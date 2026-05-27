from loguru import logger
from pyzotero import zotero
from omegaconf import DictConfig, ListConfig
from .utils import glob_match
from .retriever import get_retriever_cls
from .protocol import CorpusPaper
import random
from datetime import datetime
from .reranker import get_reranker_cls
from .construct_email import render_email, render_telegram_messages
from .utils import send_email, send_telegram
from openai import OpenAI
from tqdm import tqdm


SUPPORTED_NOTIFIERS = ("telegram", "email")


def _missing_required_fields(section, required: list[str]) -> list[str]:
    missing = []
    for key in required:
        value = getattr(section, key, None)
        if value is None or (isinstance(value, str) and value.strip() == ""):
            missing.append(key)
    return missing


def validate_notifier_config(config: DictConfig) -> str:
    """Return the active notifier name after validating its required fields.

    Raises ``ValueError`` when the selected notifier is unknown or missing
    required configuration. The check is intentionally per-notifier so users
    only need to supply credentials for the channel they actually use.
    """
    notifier = getattr(config.executor, "notifier", "telegram") or "telegram"
    notifier = notifier.lower()
    if notifier not in SUPPORTED_NOTIFIERS:
        raise ValueError(
            f"Unknown executor.notifier '{notifier}'. Supported values: {SUPPORTED_NOTIFIERS}."
        )

    if notifier == "telegram":
        missing = _missing_required_fields(config.telegram, ["bot_token", "chat_id"])
        if missing:
            raise ValueError(
                "executor.notifier=telegram requires telegram."
                + ", telegram.".join(missing)
                + ". Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID secrets, or switch executor.notifier to 'email'."
            )
    else:
        missing = _missing_required_fields(
            config.email,
            ["sender", "receiver", "smtp_server", "smtp_port", "sender_password"],
        )
        if missing:
            raise ValueError(
                "executor.notifier=email requires email."
                + ", email.".join(missing)
                + ". Set the SENDER/RECEIVER/SENDER_PASSWORD/SMTP secrets, or switch executor.notifier to 'telegram'."
            )
    return notifier


def normalize_path_patterns(patterns: list[str] | ListConfig | None, config_key: str) -> list[str] | None:
    if patterns is None:
        return None

    if not isinstance(patterns, (list, ListConfig)):
        raise TypeError(
            f"config.zotero.{config_key} must be a list of glob patterns or null, "
            'for example ["2026/survey/**"]. Single strings are not supported.'
        )

    if any(not isinstance(pattern, str) for pattern in patterns):
        raise TypeError(f"config.zotero.{config_key} must contain only glob pattern strings.")

    return list(patterns)


class Executor:
    def __init__(self, config:DictConfig):
        self.config = config
        self.notifier = validate_notifier_config(config)
        self.include_path_patterns = normalize_path_patterns(config.zotero.include_path, "include_path")
        self.ignore_path_patterns = normalize_path_patterns(config.zotero.ignore_path, "ignore_path")
        self.retrievers = {
            source: get_retriever_cls(source)(config) for source in config.executor.source
        }
        self.reranker = get_reranker_cls(config.executor.reranker)(config)
        self.openai_client = OpenAI(api_key=config.llm.api.key, base_url=config.llm.api.base_url)
    def fetch_zotero_corpus(self) -> list[CorpusPaper]:
        logger.info("Fetching zotero corpus")
        zot = zotero.Zotero(self.config.zotero.user_id, 'user', self.config.zotero.api_key)
        collections = zot.everything(zot.collections())
        collections = {c['key']:c for c in collections}
        corpus = zot.everything(zot.items(itemType='conferencePaper || journalArticle || preprint'))
        corpus = [c for c in corpus if c['data']['abstractNote'] != '']
        def get_collection_path(col_key:str) -> str:
            if p := collections[col_key]['data']['parentCollection']:
                return get_collection_path(p) + '/' + collections[col_key]['data']['name']
            else:
                return collections[col_key]['data']['name']
        for c in corpus:
            paths = [get_collection_path(col) for col in c['data']['collections']]
            c['paths'] = paths
        logger.info(f"Fetched {len(corpus)} zotero papers")
        return [CorpusPaper(
            title=c['data']['title'],
            abstract=c['data']['abstractNote'],
            added_date=datetime.strptime(c['data']['dateAdded'], '%Y-%m-%dT%H:%M:%SZ'),
            paths=c['paths']
        ) for c in corpus]
    
    def filter_corpus(self, corpus:list[CorpusPaper]) -> list[CorpusPaper]:
        if self.include_path_patterns:
            logger.info(f"Selecting zotero papers matching include_path: {self.include_path_patterns}")
            corpus = [
                c for c in corpus
                if any(
                    glob_match(path, pattern)
                    for path in c.paths
                    for pattern in self.include_path_patterns
                )
            ]
        if self.ignore_path_patterns:
            logger.info(f"Excluding zotero papers matching ignore_path: {self.ignore_path_patterns}")
            corpus = [
                c for c in corpus
                if not any(
                    glob_match(path, pattern)
                    for path in c.paths
                    for pattern in self.ignore_path_patterns
                )
            ]
        if self.include_path_patterns or self.ignore_path_patterns:
            samples = random.sample(corpus, min(5, len(corpus)))
            samples = '\n'.join([c.title + ' - ' + '\n'.join(c.paths) for c in samples])
            logger.info(f"Selected {len(corpus)} zotero papers:\n{samples}\n...")
        return corpus

    
    def run(self):
        corpus = self.fetch_zotero_corpus()
        corpus = self.filter_corpus(corpus)
        if len(corpus) == 0:
            logger.error(f"No zotero papers found. Please check your zotero settings:\n{self.config.zotero}")
            return
        all_papers = []
        for source, retriever in self.retrievers.items():
            logger.info(f"Retrieving {source} papers...")
            papers = retriever.retrieve_papers()
            if len(papers) == 0:
                logger.info(f"No {source} papers found")
                continue
            logger.info(f"Retrieved {len(papers)} {source} papers")
            all_papers.extend(papers)
        logger.info(f"Total {len(all_papers)} papers retrieved from all sources")
        reranked_papers = []
        if len(all_papers) > 0:
            logger.info("Reranking papers...")
            reranked_papers = self.reranker.rerank(all_papers, corpus)
            reranked_papers = reranked_papers[:self.config.executor.max_paper_num]
            logger.info("Generating TLDR and affiliations...")
            for p in tqdm(reranked_papers):
                p.generate_tldr(self.openai_client, self.config.llm)
                p.generate_affiliations(self.openai_client, self.config.llm)
        elif not self.config.executor.send_empty:
            logger.info(f"No new papers found. No {self.notifier} notification will be sent.")
            return
        logger.info(f"Sending notification via {self.notifier}...")
        if self.notifier == "telegram":
            messages = render_telegram_messages(reranked_papers)
            send_telegram(self.config, messages)
        else:
            email_content = render_email(reranked_papers)
            send_email(self.config, email_content)
        logger.info(f"{self.notifier.capitalize()} notification sent successfully")
