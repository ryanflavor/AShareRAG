"""Vector Retriever component for fact-based Q&A."""

import logging
import re
import time
from typing import Any

from src.components.embedding_service import EmbeddingService
from src.components.vector_storage import VectorStorage

logger = logging.getLogger(__name__)


class VectorRetriever:
    """Retrieves relevant documents from vector storage based on query similarity."""

    def __init__(
        self,
        vector_storage: VectorStorage,
        embedding_service: EmbeddingService,
        top_k: int = 10,
    ):
        """Initialize the VectorRetriever.

        Args:
            vector_storage: Vector storage instance for document retrieval
            embedding_service: Service for generating query embeddings
            top_k: Number of top results to retrieve (default: 10)
        """
        self.vector_storage = vector_storage
        self.embedding_service = embedding_service
        self.top_k = top_k

        # Statistics
        self._total_queries = 0
        self._total_retrieval_time = 0.0

    def retrieve(
        self, query: str, company_filter: str | None = None, top_k: int | None = None
    ) -> list[dict[str, Any]]:
        """Retrieve relevant documents for the given query.

        Args:
            query: User query in natural language
            company_filter: Optional company name to filter results
            top_k: Optional override for number of results to retrieve

        Returns:
            List of retrieved documents with content, metadata, and scores

        Raises:
            ValueError: If query is empty or exceeds maximum length
        """
        # Input validation
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        # Sanitize query - remove special characters while preserving Chinese
        query = re.sub(r'[^\w\s\u4e00-\u9fff\u3000-\u303f\uff00-\uffef？！。，、；：""''（）《》【】+*/=\\?-]', '', query)
        query = query.strip()

        if not query:
            raise ValueError("Query cannot be empty after sanitization")

        # Check query length
        if len(query) > 10000:
            raise ValueError(f"Query exceeds maximum length of 10000 characters (got {len(query)})")

        # Use provided top_k or default
        retrieval_top_k = top_k or self.top_k

        # Log start time
        start_time = time.time()

        # Generate query embedding
        embedding_start = time.time()
        query_embeddings = self.embedding_service.generate_embeddings([query])
        query_embedding = query_embeddings[0]
        embedding_time = time.time() - embedding_start
        logger.info(f"Embedding generation time: {embedding_time:.3f}s")

        # Auto-extract company if not provided
        if not company_filter:
            company_filter = self.extract_company_from_query(query)
            if company_filter:
                logger.info(f"Auto-extracted company: {company_filter}")

        # Search vector storage
        search_start = time.time()
        search_kwargs = {"query_vector": query_embedding, "top_k": retrieval_top_k}

        if company_filter:
            search_kwargs["filter_company"] = company_filter

        results = self.vector_storage.search(**search_kwargs)
        search_time = time.time() - search_start
        logger.info(f"Vector search time: {search_time:.3f}s")

        # Format results with deduplication
        formatted_results = []
        seen_contents = set()

        for result in results[: retrieval_top_k]:  # Ensure top_k limit
            # Get content from 'text' field (database schema) or 'content' field (fallback)
            content = result.get("text", result.get("content", ""))

            # Skip duplicates
            if content in seen_contents:
                continue

            seen_contents.add(content)

            formatted_result = {
                "content": content,
                "company": result.get("company_name", ""),
                "score": result.get("similarity", 0.0),
                "metadata": result.get("metadata", {}),
            }
            formatted_results.append(formatted_result)

        # Sort by score descending
        formatted_results.sort(key=lambda x: x["score"], reverse=True)

        total_time = time.time() - start_time
        logger.info(
            f"Total retrieval time: {total_time:.3f}s, "
            f"retrieved {len(formatted_results)} documents"
        )

        # Update statistics
        self._total_queries += 1
        self._total_retrieval_time += total_time

        return formatted_results

    def extract_company_from_query(self, query: str) -> str | None:
        """Extract company name from query text.

        Args:
            query: User query text

        Returns:
            Extracted company name or None if not found
        """
        # Common A-share company patterns
        # This is a simple implementation - could be enhanced with NER
        company_patterns = [
            r"(贵州茅台|茅台)",
            r"(比亚迪)",
            r"(宁德时代)",
            r"(中国平安|平安)",
            r"(招商银行|招行)",
            r"(五粮液)",
            r"(中国移动|移动)",
            r"(长江电力)",
            r"(紫金矿业)",
            r"(中国石油|中石油)",
            r"(中国人寿|人寿)",
            r"(中国石化|中石化)",
            r"(农业银行|农行)",
            r"(中国银行|中行)",
            r"(工商银行|工行)",
            r"(建设银行|建行)",
            r"(中国联通|联通)",
            r"(中国电信|电信)",
            r"(中国神华|神华)",
            r"(海螺水泥)",
            r"(万科A|万科)",
            r"(美的集团|美的)",
            r"(格力电器|格力)",
            r"(恒瑞医药|恒瑞)",
            r"(海天味业|海天)",
            r"(中国中免|中免)",
            r"(山西汾酒|汾酒)",
            r"(泸州老窖|老窖)",
            r"(中国太保|太保)",
            r"(中信证券|中信)",
            r"(隆基绿能|隆基)",
            r"(三一重工|三一)",
            r"(中国建筑|中建)",
            r"(中国交建|交建)",
            r"(京东方A|京东方)",
            r"(TCL科技|TCL)",
            r"(长城汽车|长城)",
            r"(伊利股份|伊利)",
            r"(顺丰控股|顺丰)",
            r"(牧原股份|牧原)",
            r"(通威股份|通威)",
            r"(东方财富)",
            r"(片仔癀)",
            r"(智飞生物|智飞)",
            r"(爱尔眼科|爱尔)",
            r"(药明康德|药明)",
            r"(恒生电子|恒生)",
            r"(中芯国际|中芯)",
            r"(韦尔股份|韦尔)",
            r"(立讯精密|立讯)",
            r"(汇川技术|汇川)",
            r"(迈瑞医疗|迈瑞)",
            r"(阳光电源|阳光)",
            r"(金龙鱼)",
            r"(海康威视|海康)",
            r"(大华股份|大华)",
            r"(科大讯飞|讯飞)",
            r"(浪潮信息|浪潮)",
            r"(中兴通讯|中兴)",
            r"(北方华创|华创)",
            r"(兆易创新|兆易)",
            r"(闻泰科技|闻泰)",
            r"(紫光国微|紫光)",
            r"(卓胜微)",
            r"(圣邦股份|圣邦)",
            r"(南京银行|南京)",
            r"(兴业银行|兴业)",
            r"(浦发银行|浦发)",
            r"(民生银行|民生)",
            r"(光大银行|光大)",
            r"(华夏银行|华夏)",
            r"(北京银行|北京)",
            r"(上海银行|上海)",
            r"(江苏银行|江苏)",
            r"(宁波银行|宁波)",
            r"(杭州银行|杭州)",
            r"(成都银行|成都)",
            r"(长沙银行|长沙)",
            r"(青岛银行|青岛)",
            r"(苏州银行|苏州)",
            r"(郑州银行|郑州)",
            r"(西安银行|西安)",
            r"(重庆银行|重庆)",
            r"(贵阳银行|贵阳)",
            r"(徽商银行|徽商)",
            r"(天津银行|天津)",
            r"(锦州银行|锦州)",
            r"(威海银行|威海)",
            r"(甘肃银行|甘肃)",
            r"(石药集团|石药)",
            r"(复星医药|复星)",
            r"(上海医药|上药)",
            r"(白云山)",
            r"(同仁堂)",
            r"(云南白药|白药)",
            r"(康美药业|康美)",
            r"(东阿阿胶|阿胶)",
            r"(华润三九|三九)",
            r"(丽珠集团|丽珠)",
            r"(华兰生物|华兰)",
            r"(长春高新|高新)",
            r"(沃森生物|沃森)",
            r"(康泰生物|康泰)",
            r"(万泰生物|万泰)",
            r"(安科生物|安科)",
            r"(华大基因|华大)",
            r"(贝达药业|贝达)",
            r"(信达生物|信达)",
            r"(君实生物|君实)",
            r"(百济神州|百济)",
            r"(泰格医药|泰格)",
            r"(凯莱英)",
            r"(昭衍新药|昭衍)",
            r"(康龙化成|康龙)",
            r"(美迪西)",
            r"(博腾股份|博腾)",
        ]

        # Try to match company patterns
        for pattern in company_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                # Return the full company name (first group in pattern)
                full_match = match.group(0)
                # Map abbreviated names to full names
                name_mapping = {
                    "茅台": "贵州茅台",
                    "平安": "中国平安",
                    "招行": "招商银行",
                    "移动": "中国移动",
                    "中石油": "中国石油",
                    "人寿": "中国人寿",
                    "中石化": "中国石化",
                    "农行": "农业银行",
                    "中行": "中国银行",
                    "工行": "工商银行",
                    "建行": "建设银行",
                    "联通": "中国联通",
                    "电信": "中国电信",
                    "神华": "中国神华",
                    "万科": "万科A",
                    "美的": "美的集团",
                    "格力": "格力电器",
                    "恒瑞": "恒瑞医药",
                    "海天": "海天味业",
                    "中免": "中国中免",
                    "汾酒": "山西汾酒",
                    "老窖": "泸州老窖",
                    "太保": "中国太保",
                    "中信": "中信证券",
                    "隆基": "隆基绿能",
                    "三一": "三一重工",
                    "中建": "中国建筑",
                    "交建": "中国交建",
                    "京东方": "京东方A",
                    "TCL": "TCL科技",
                    "长城": "长城汽车",
                    "伊利": "伊利股份",
                    "顺丰": "顺丰控股",
                    "牧原": "牧原股份",
                    "通威": "通威股份",
                    "智飞": "智飞生物",
                    "爱尔": "爱尔眼科",
                    "药明": "药明康德",
                    "恒生": "恒生电子",
                    "中芯": "中芯国际",
                    "韦尔": "韦尔股份",
                    "立讯": "立讯精密",
                    "汇川": "汇川技术",
                    "迈瑞": "迈瑞医疗",
                    "阳光": "阳光电源",
                    "海康": "海康威视",
                    "大华": "大华股份",
                    "讯飞": "科大讯飞",
                    "浪潮": "浪潮信息",
                    "中兴": "中兴通讯",
                    "华创": "北方华创",
                    "兆易": "兆易创新",
                    "闻泰": "闻泰科技",
                    "紫光": "紫光国微",
                    "圣邦": "圣邦股份",
                    "南京": "南京银行",
                    "兴业": "兴业银行",
                    "浦发": "浦发银行",
                    "民生": "民生银行",
                    "光大": "光大银行",
                    "华夏": "华夏银行",
                    "北京": "北京银行",
                    "上海": "上海银行",
                    "江苏": "江苏银行",
                    "宁波": "宁波银行",
                    "杭州": "杭州银行",
                    "成都": "成都银行",
                    "长沙": "长沙银行",
                    "青岛": "青岛银行",
                    "苏州": "苏州银行",
                    "郑州": "郑州银行",
                    "西安": "西安银行",
                    "重庆": "重庆银行",
                    "贵阳": "贵阳银行",
                    "徽商": "徽商银行",
                    "天津": "天津银行",
                    "锦州": "锦州银行",
                    "威海": "威海银行",
                    "甘肃": "甘肃银行",
                    "石药": "石药集团",
                    "复星": "复星医药",
                    "上药": "上海医药",
                    "白药": "云南白药",
                    "康美": "康美药业",
                    "阿胶": "东阿阿胶",
                    "三九": "华润三九",
                    "丽珠": "丽珠集团",
                    "华兰": "华兰生物",
                    "高新": "长春高新",
                    "沃森": "沃森生物",
                    "康泰": "康泰生物",
                    "万泰": "万泰生物",
                    "安科": "安科生物",
                    "华大": "华大基因",
                    "贝达": "贝达药业",
                    "信达": "信达生物",
                    "君实": "君实生物",
                    "百济": "百济神州",
                    "泰格": "泰格医药",
                    "昭衍": "昭衍新药",
                    "康龙": "康龙化成",
                    "博腾": "博腾股份",
                }

                # Return mapped name or original match
                return name_mapping.get(full_match, full_match)

        return None

    def get_statistics(self) -> dict[str, Any]:
        """Get retrieval statistics.

        Returns:
            Dictionary with statistics
        """
        avg_time = (
            self._total_retrieval_time / self._total_queries
            if self._total_queries > 0
            else 0
        )

        return {
            "total_queries": self._total_queries,
            "average_retrieval_time": avg_time,
        }
