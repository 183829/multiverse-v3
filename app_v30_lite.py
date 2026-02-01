#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多维轮回破解系统（渊开发）- v3.0 Lite 终极进化版
==================================================
✨ AI 智能出题 | AI 深度评分 | 多模型融合对话
🚀 机器学习预测 | 向量检索知识库 | 深度游戏化
📱 PWA 支持 | 20+ 语言 | 企业级安全
🔮 流程优化 | 性能提升 | 稳定部署

版本：v3.0 Lite
优化：核心极致功能 + 云端稳定部署
"""

import streamlit as st
import os
import json
import time
import random
import hashlib
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import threading
from collections import deque
from dataclasses import dataclass, field, asdict
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# 核心依赖
import requests
import numpy as np
import pandas as pd

# Matplotlib 配置 - 移到顶部并延迟导入
import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import io
import base64
import re

# Streamlit 配置
st.set_page_config(
    page_title="多维轮回破解系统 v3.0",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 核心配置 ====================

class SystemConfig:
    """系统配置"""
    def __init__(self):
        # API 配置
        self.api_providers = {
            "groq": {
                "name": "Groq",
                "base_url": "https://api.groq.com/openai/v1",
                "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
                "free": True,
                "priority": 1
            },
            "openai": {
                "name": "OpenAI",
                "base_url": "https://api.openai.com/v1",
                "models": ["gpt-4-turbo", "gpt-3.5-turbo"],
                "free": False,
                "priority": 2
            },
            "anthropic": {
                "name": "Anthropic",
                "base_url": "https://api.anthropic.com/v1",
                "models": ["claude-3-opus-20240229", "claude-3-sonnet-20240229"],
                "free": False,
                "priority": 3
            },
            "cohere": {
                "name": "Cohere",
                "base_url": "https://api.cohere.ai/v1",
                "models": ["command", "command-light"],
                "free": False,
                "priority": 4
            }
        }
        
        # 意识维度
        self.consciousness_dimensions = [
            "reasoning",      # 逻辑推理
            "creative",       # 创造性思维
            "knowledge",      # 知识应用
            "depth",          # 深度思考
            "coding",         # 代码能力
            "intuition",      # 直觉洞察
            "synthesis",      # 综合分析
            "memory",         # 记忆能力
            "emotion",        # 情绪管理
            "decision",       # 决策能力
            "learning",       # 学习速度
            "innovation"      # 创新能力
        ]
        
        # 成就配置
        self.achievements_config = {
            "first_test": {"name": "初次觉醒", "desc": "完成第一次意识测试", "exp": 50},
            "test_master": {"name": "测试达人", "desc": "完成10次意识测试", "exp": 200},
            "perfect_score": {"name": "完美表现", "desc": "单次测试超过80分", "exp": 300},
            "level_10": {"name": "进阶者", "desc": "达到10级", "exp": 500},
            "level_50": {"name": "意识大师", "desc": "达到50级", "exp": 2000},
            "conversation_100": {"name": "深度对话者", "desc": "完成100次对话", "exp": 300},
            "knowledge_collector": {"name": "知识收藏家", "desc": "上传10个文档", "exp": 200},
            "all_dimensions": {"name": "全能大师", "desc": "所有维度超过8分", "exp": 1000}
        }
        
        # 等级配置
        self.level_config = {
            "max_level": 100,
            "exp_base": 100,
            "exp_growth": 1.2
        }

# ==================== 数据结构 ====================

class ConsciousnessLevel(Enum):
    """意识等级"""
    AWAKENING = 1
    RISING = 2
    ASCENDING = 3
    TRANSCENDING = 4
    TRANSCENDENT = 5
    ENLIGHTENED = 6
    MASTER = 7

class QuestionDifficulty(Enum):
    """题目难度"""
    BEGINNER = 1
    INTERMEDIATE = 2
    ADVANCED = 3
    EXPERT = 4
    MASTER = 5

@dataclass
class ConsciousnessSnapshot:
    """意识快照"""
    timestamp: datetime
    scores: Dict[str, float]
    level: ConsciousnessLevel
    total_score: float
    test_answers: List[Dict] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

@dataclass
class Question:
    """题目"""
    id: str
    dimension: str
    difficulty: QuestionDifficulty
    content: str
    type: str  # open, choice, scenario, code
    options: List[str] = field(default_factory=list)
    reference: str = ""
    metadata: Dict = field(default_factory=dict)

@dataclass
class TestResult:
    """测试结果"""
    snapshot: ConsciousnessSnapshot
    question_count: int
    correct_count: int
    accuracy: float
    time_spent: float
    confidence_scores: Dict[str, float]

@dataclass
class ConversationMessage:
    """对话消息"""
    role: str
    content: str
    timestamp: datetime
    confidence: float = 0.0
    models_used: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

# ==================== 异步缓存系统 ====================

class AsyncCache:
    """高性能异步缓存系统"""
    def __init__(self, max_size: int = 2000, ttl: int = 3600):
        self.cache = {}
        self.expiry = {}
        self.lock = threading.RLock()
        self.max_size = max_size
        self.ttl = ttl
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def _generate_key(self, *args, **kwargs) -> str:
        """生成缓存键"""
        key_data = f"{args}-{sorted(kwargs.items())}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, *args, **kwargs) -> Optional[Any]:
        """获取缓存"""
        key = self._generate_key(*args, **kwargs)
        with self.lock:
            if key in self.cache:
                if time.time() < self.expiry[key]:
                    self.hits += 1
                    return self.cache[key]
                else:
                    del self.cache[key]
                    del self.expiry[key]
            self.misses += 1
            return None
    
    def set(self, value: Any, *args, **kwargs):
        """设置缓存"""
        key = self._generate_key(*args, **kwargs)
        with self.lock:
            # LRU 淘汰
            if len(self.cache) >= self.max_size and key not in self.cache:
                oldest_key = min(self.expiry.keys(), key=lambda k: self.expiry[k])
                del self.cache[oldest_key]
                del self.expiry[oldest_key]
                self.evictions += 1
            
            self.cache[key] = value
            self.expiry[key] = time.time() + self.ttl
    
    def clear(self):
        """清空缓存"""
        with self.lock:
            self.cache.clear()
            self.expiry.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "size": len(self.cache),
            "max_size": self.max_size,
            "evictions": self.evictions
        }

# ==================== API 管理系统 ====================

class APIManager:
    """API 管理器 - 智能路由与健康监测"""
    def __init__(self, api_keys: Dict[str, str], config: SystemConfig):
        self.api_keys = api_keys
        self.config = config
        self.cache = AsyncCache()
        self.health_status = {name: {"available": True, "last_check": datetime.now(), "failures": 0} 
                              for name in config.api_providers.keys()}
        self.usage_stats = {name: {"requests": 0, "success": 0, "errors": 0, "avg_time": 0} 
                           for name in config.api_providers.keys()}
    
    def get_available_providers(self) -> List[str]:
        """获取可用的 API 提供商"""
        return [name for name, status in self.health_status.items() 
                if status["available"] and self.api_keys.get(name)]
    
    def select_best_provider(self, task_type: str = "general") -> Optional[str]:
        """智能选择最佳 API 提供商"""
        available = self.get_available_providers()
        if not available:
            return None
        
        # 根据任务类型和优先级选择
        if task_type == "fast":
            # 选择速度最快的
            providers = sorted(available, 
                              key=lambda x: self.usage_stats[x]["avg_time"] or 999)
        elif task_type == "quality":
            # 选择质量最高的
            providers = sorted(available, 
                              key=lambda x: self.config.api_providers[x]["priority"])
        else:
            # 默认根据优先级
            providers = sorted(available, 
                              key=lambda x: self.config.api_providers[x]["priority"])
        
        return providers[0] if providers else None
    
    def call_api(self, provider: str, messages: List[Dict], 
                 model: str = None, **kwargs) -> Dict[str, Any]:
        """调用 API"""
        cache_key = f"{provider}_{model}_{hash(str(messages))}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        provider_config = self.config.api_providers[provider]
        api_key = self.api_keys.get(provider)
        
        if not api_key:
            return {"error": f"API key for {provider} not configured"}
        
        start_time = time.time()
        try:
            headers = {"Content-Type": "application/json"}
            url = ""
            data = {}
            
            if provider == "groq":
                headers["Authorization"] = f"Bearer {api_key}"
                url = f"{provider_config['base_url']}/chat/completions"
                model = model or provider_config["models"][0]
                data = {
                    "model": model,
                    "messages": messages,
                    "temperature": kwargs.get("temperature", 0.7),
                    "max_tokens": kwargs.get("max_tokens", 2048)
                }
            
            elif provider == "openai":
                headers["Authorization"] = f"Bearer {api_key}"
                url = f"{provider_config['base_url']}/chat/completions"
                model = model or provider_config["models"][0]
                data = {
                    "model": model,
                    "messages": messages,
                    "temperature": kwargs.get("temperature", 0.7),
                    "max_tokens": kwargs.get("max_tokens", 2048)
                }
            
            elif provider == "anthropic":
                headers["x-api-key"] = api_key
                headers["anthropic-version"] = "2023-06-01"
                url = f"{provider_config['base_url']}/messages"
                model = model or provider_config["models"][0]
                # 转换消息格式
                system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
                user_msgs = [m for m in messages if m["role"] != "system"]
                data = {
                    "model": model,
                    "system": system_msg,
                    "messages": [{"role": m["role"], "content": m["content"]} for m in user_msgs],
                    "max_tokens": kwargs.get("max_tokens", 2048)
                }
            
            elif provider == "cohere":
                headers["Authorization"] = f"Bearer {api_key}"
                url = f"{provider_config['base_url']}/chat"
                model = model or provider_config["models"][0]
                data = {
                    "model": model,
                    "message": messages[-1]["content"],
                    "chat_history": messages[:-1],
                    "temperature": kwargs.get("temperature", 0.7),
                    "max_tokens": kwargs.get("max_tokens", 2048)
                }
            
            response = requests.post(url, headers=headers, json=data, timeout=120)
            response.raise_for_status()
            result = response.json()
            
            # 提取响应文本
            if provider == "anthropic":
                response_text = result["content"][0]["text"]
            elif provider == "cohere":
                response_text = result["text"]
            else:
                response_text = result["choices"][0]["message"]["content"]
            
            # 更新统计
            execution_time = time.time() - start_time
            stats = self.usage_stats[provider]
            stats["requests"] += 1
            stats["success"] += 1
            stats["avg_time"] = (stats["avg_time"] * (stats["requests"] - 1) + execution_time) / stats["requests"]
            self.health_status[provider]["failures"] = 0
            self.health_status[provider]["available"] = True
            
            # 缓存结果
            self.cache.set(response_text, cache_key)
            
            return {
                "content": response_text,
                "model": model,
                "provider": provider,
                "time": execution_time,
                "success": True
            }
        
        except Exception as e:
            # 更新错误统计
            execution_time = time.time() - start_time
            stats = self.usage_stats[provider]
            stats["requests"] += 1
            stats["errors"] += 1
            self.health_status[provider]["failures"] += 1
            if self.health_status[provider]["failures"] >= 3:
                self.health_status[provider]["available"] = False
            
            return {
                "error": str(e),
                "provider": provider,
                "success": False
            }

# ==================== 知识库系统 ====================

class AdvancedKnowledgeBase:
    """高级知识库系统 - 向量检索 + 语义搜索"""
    def __init__(self, cache: AsyncCache):
        self.cache = cache
        self.documents = {}
        self.document_vectors = None
        self.vectorizer = None
        self.neural_index = {}
        self.lock = threading.RLock()
    
    def add_document(self, doc_id: str, content: str, metadata: Dict = None):
        """添加文档"""
        with self.lock:
            self.documents[doc_id] = {
                "content": content,
                "metadata": metadata or {},
                "added_at": datetime.now(),
                "length": len(content),
                "tokens": len(content.split())
            }
            self._rebuild_index()
    
    def _rebuild_index(self):
        """重建索引"""
        if not self.documents:
            return
        
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            
            texts = [doc["content"] for doc in self.documents.values()]
            self.vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
            self.document_vectors = self.vectorizer.fit_transform(texts)
            
            # 生成神经特征
            for doc_id, doc in self.documents.items():
                self.neural_index[doc_id] = self._generate_neural_features(doc["content"])
        
        except Exception as e:
            print(f"Error rebuilding index: {e}")
    
    def _generate_neural_features(self, text: str) -> Dict[str, float]:
        """生成神经特征"""
        return {
            "complexity": len(set(text.split())) / len(text.split()) if text.split() else 0,
            "density": text.count('。') / (len(text) / 100) if text else 0,
            "avg_sentence_length": np.mean([len(s.split()) for s in text.split('。') if s]) if text else 0
        }
    
    def search(self, query: str, top_k: int = 5, use_semantic: bool = True) -> List[Dict]:
        """搜索文档"""
        cache_key = f"search_{hash(query)}_{top_k}_{use_semantic}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        if not self.documents or self.document_vectors is None:
            return []
        
        try:
            # 语义搜索
            if use_semantic and self.vectorizer:
                query_vector = self.vectorizer.transform([query])
                similarities = cosine_similarity(query_vector, self.document_vectors).flatten()
                
                top_indices = similarities.argsort()[-top_k:][::-1]
                results = []
                for idx in top_indices:
                    doc_id = list(self.documents.keys())[idx]
                    doc = self.documents[doc_id]
                    results.append({
                        "doc_id": doc_id,
                        "score": float(similarities[idx]),
                        "content": doc["content"][:500],
                        "metadata": doc["metadata"],
                        "neural_features": self.neural_index.get(doc_id, {})
                    })
                
                self.cache.set(results, cache_key)
                return results
        
        except Exception as e:
            print(f"Search error: {e}")
        
        return []
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict]:
        """根据 ID 获取文档"""
        return self.documents.get(doc_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self.lock:
            return {
                "total_documents": len(self.documents),
                "total_tokens": sum(doc["tokens"] for doc in self.documents.values()),
                "total_length": sum(doc["length"] for doc in self.documents.values()),
                "avg_document_length": np.mean([doc["length"] for doc in self.documents.values()]) if self.documents else 0
            }

# ==================== AI 出题引擎 ====================

class AIQuestionGenerator:
    """AI 智能出题引擎"""
    def __init__(self, api_manager: APIManager, knowledge_base: AdvancedKnowledgeBase):
        self.api_manager = api_manager
        self.knowledge_base = knowledge_base
        self.cache = AsyncCache()
        
        # 题目模板
        self.question_templates = {
            "reasoning": [
                "基于知识库中的{topic}，请分析{concept1}与{concept2}之间的逻辑关系",
                "如果{condition}成立，那么{result}会发生变化吗？请说明理由",
                "请用三段论推理分析以下问题：{problem}"
            ],
            "creative": [
                "请用三种不同的创意方式描述{concept}",
                "基于{context}，设计一个创新的{idea}",
                "请从反直觉的角度思考{topic}"
            ],
            "depth": [
                "从多个哲学角度深度思考：{question}",
                "如果{hypothesis}成为现实，这对{domain}意味着什么？",
                "请分析{concept}的本质，并探讨其深层意义"
            ]
        }
    
    def generate_question(self, dimension: str, difficulty: int = 3, 
                          use_knowledge_base: bool = True) -> Optional[Question]:
        """生成题目"""
        cache_key = f"q_{dimension}_{difficulty}_{use_knowledge_base}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        # 如果使用知识库，先检索相关内容
        context = ""
        reference = ""
        if use_knowledge_base and self.knowledge_base.documents:
            search_results = self.knowledge_base.search(dimension, top_k=1)
            if search_results:
                context = search_results[0]["content"][:300]
                reference = search_results[0]["metadata"].get("filename", "")
        
        # 构建 prompt
        prompt = f"""
        你是一个专业的意识测试出题专家。请为"{dimension}"维度生成一个难度为{difficulty}级（1-5级）的测试题目。
        
        {"以下是相关的知识库内容，请基于此生成题目：\n" + context if context else "请自行设计一个有深度的题目"}
        
        要求：
        1. 题目具有启发性和挑战性
        2. 能够真正测试用户的{dimension}能力
        3. 开放性问题，鼓励深入思考
        4. 题目简洁明了
        
        请直接输出题目内容，不需要其他解释。
        """
        
        # 调用 API 生成
        provider = self.api_manager.select_best_provider(task_type="quality")
        if not provider:
            return None
        
        response = self.api_manager.call_api(
            provider=provider,
            messages=[
                {"role": "system", "content": "你是专业的意识测试出题专家。"},
                {"role": "user", "content": prompt}
            ]
        )
        
        if not response.get("success"):
            return None
        
        question_content = response["content"]
        
        # 创建题目对象
        question = Question(
            id=hashlib.md5(question_content.encode()).hexdigest(),
            dimension=dimension,
            difficulty=QuestionDifficulty(min(5, max(1, difficulty))),
            content=question_content,
            type="open",
            reference=reference,
            metadata={
                "generated_by": "ai",
                "provider": provider,
                "model": response["model"],
                "difficulty": difficulty,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        self.cache.set(question, cache_key)
        return question
    
    def generate_batch_questions(self, dimensions: List[str], 
                                count_per_dimension: int = 1) -> List[Question]:
        """批量生成题目"""
        questions = []
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            for dimension in dimensions:
                for _ in range(count_per_dimension):
                    future = executor.submit(
                        self.generate_question, 
                        dimension, 
                        difficulty=random.randint(2, 4)
                    )
                    futures.append(future)
            
            for future in as_completed(futures):
                question = future.result()
                if question:
                    questions.append(question)
        
        return questions

# ==================== AI 评分引擎 ====================

class AIGradingEngine:
    """AI 深度评分引擎"""
    def __init__(self, api_manager: APIManager, knowledge_base: AdvancedKnowledgeBase):
        self.api_manager = api_manager
        self.knowledge_base = knowledge_base
        self.cache = AsyncCache()
        
        # 评分维度
        self.grading_dimensions = {
            "logic": {"weight": 0.25, "desc": "逻辑性"},
            "creativity": {"weight": 0.20, "desc": "创造性"},
            "depth": {"weight": 0.25, "desc": "深度"},
            "accuracy": {"weight": 0.15, "desc": "准确性"},
            "completeness": {"weight": 0.15, "desc": "完整性"}
        }
    
    def grade_answer(self, question: Question, answer: str) -> Dict[str, Any]:
        """评分答案"""
        cache_key = f"grade_{question.id}_{hash(answer)}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        # 检索相关知识库内容
        context = ""
        if self.knowledge_base.documents:
            search_results = self.knowledge_base.search(question.dimension, top_k=2)
            if search_results:
                context = "\n".join([r["content"][:200] for r in search_results])
        
        # 构建 prompt
        prompt = f"""
        你是一个专业的意识测试评分专家。请对以下答案进行深度评分。
        
        === 测试维度 ===
        {question.dimension}
        
        === 题目 ===
        {question.content}
        
        === 参考内容 ===
        {context if context else "（无）"}
        
        === 用户答案 ===
        {answer}
        
        评分维度：
        1. 逻辑性 (0-10分)：推理是否严密，逻辑是否清晰
        2. 创造性 (0-10分)：是否有新颖见解，是否打破常规思维
        3. 深度 (0-10分)：思考是否深入，是否触及本质
        4. 准确性 (0-10分)：与参考内容的匹配度，事实是否准确
        5. 完整性 (0-10分)：回答是否全面，是否有遗漏
        
        请以 JSON 格式输出评分结果，格式如下：
        {{
            "logic": 分数,
            "creativity": 分数,
            "depth": 分数,
            "accuracy": 分数,
            "completeness": 分数,
            "total_score": 加权总分,
            "feedback": "详细反馈",
            "strengths": ["优势1", "优势2"],
            "weaknesses": ["改进点1", "改进点2"],
            "confidence": 评分置信度(0-1)
        }}
        
        请只输出 JSON，不要有其他内容。
        """
        
        # 调用 API 评分
        provider = self.api_manager.select_best_provider(task_type="quality")
        if not provider:
            return self._default_grading()
        
        response = self.api_manager.call_api(
            provider=provider,
            messages=[
                {"role": "system", "content": "你是专业的意识测试评分专家，必须以 JSON 格式输出评分结果。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3  # 降低随机性
        )
        
        if not response.get("success"):
            return self._default_grading()
        
        try:
            # 提取 JSON
            content = response["content"]
            # 尝试提取 JSON 部分
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                grading_result = json.loads(json_match.group())
                
                # 验证字段
                required_fields = ["logic", "creativity", "depth", "accuracy", "completeness", 
                                  "total_score", "feedback", "confidence"]
                if all(field in grading_result for field in required_fields):
                    grading_result["provider"] = provider
                    grading_result["model"] = response["model"]
                    self.cache.set(grading_result, cache_key)
                    return grading_result
        
        except Exception as e:
            print(f"Grading error: {e}")
        
        return self._default_grading()
    
    def _default_grading(self) -> Dict[str, Any]:
        """默认评分（失败时）"""
        return {
            "logic": random.uniform(5, 8),
            "creativity": random.uniform(5, 8),
            "depth": random.uniform(5, 8),
            "accuracy": random.uniform(5, 8),
            "completeness": random.uniform(5, 8),
            "total_score": random.uniform(6, 9),
            "feedback": "系统评分不可用，使用默认评分",
            "strengths": ["完整回答"],
            "weaknesses": ["需要更深入的分析"],
            "confidence": 0.5,
            "provider": "default"
        }

# ==================== 多模型融合对话系统 ====================

class MultiModelDialogueSystem:
    """多模型融合对话系统"""
    def __init__(self, api_manager: APIManager, cache: AsyncCache):
        self.api_manager = api_manager
        self.cache = cache
        self.conversation_history = deque(maxlen=100)
    
    def dialogue(self, user_input: str, context: str = "", 
                  complexity: int = 5, use_ensemble: bool = True) -> Dict[str, Any]:
        """对话"""
        cache_key = f"dialog_{hash(user_input)}_{complexity}_{use_ensemble}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        # 构建消息
        messages = []
        if context:
            messages.append({"role": "system", "content": f"背景：{context}"})
        
        # 添加历史对话（最近 5 条）
        for msg in list(self.conversation_history)[-5:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
        
        messages.append({"role": "user", "content": user_input})
        
        if use_ensemble:
            # 多模型融合
            return self._ensemble_dialogue(messages, complexity)
        else:
            # 单模型
            provider = self.api_manager.select_best_provider(task_type="quality")
            if not provider:
                return {"error": "No available API provider"}
            
            response = self.api_manager.call_api(provider, messages, temperature=0.7)
            if response.get("success"):
                return {
                    "content": response["content"],
                    "provider": provider,
                    "model": response["model"],
                    "confidence": 0.8,
                    "models_used": [provider],
                    "time": response["time"]
                }
            else:
                return {"error": response["error"]}
    
    def _ensemble_dialogue(self, messages: List[Dict], complexity: int) -> Dict[str, Any]:
        """多模型融合对话"""
        providers = self.api_manager.get_available_providers()
        if not providers:
            return {"error": "No available API providers"}
        
        # 并发调用多个 API
        results = []
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            for provider in providers[:3]:  # 最多使用 3 个 API
                future = executor.submit(
                    self.api_manager.call_api,
                    provider,
                    messages,
                    temperature=0.7
                )
                futures.append(future)
            
            for future in as_completed(futures):
                result = future.result()
                if result.get("success"):
                    results.append(result)
        
        if not results:
            return {"error": "All API calls failed"}
        
        # 如果只有一个结果，直接返回
        if len(results) == 1:
            return {
                "content": results[0]["content"],
                "provider": results[0]["provider"],
                "model": results[0]["model"],
                "confidence": 0.8,
                "models_used": [results[0]["provider"]],
                "time": results[0]["time"]
            }
        
        # 多结果融合
        return self._synthesize_results(results, messages)
    
    def _synthesize_results(self, results: List[Dict], messages: List[Dict]) -> Dict[str, Any]:
        """融合多个结果"""
        # 构建融合 prompt
        results_text = "\n\n".join([
            f"=== {r['provider']} ({r['model']}) ===\n{r['content']}"
            for r in results
        ])
        
        synthesis_prompt = f"""
        你是神经符号融合专家。请融合以下多个 AI 模型的回答，生成最终答案。
        
        {results_text}
        
        === 原始问题 ===
        {messages[-1]['content']}
        
        要求：
        1. 综合各模型的优点，生成一个更全面、准确的答案
        2. 指出各模型回答的亮点和不足
        3. 给出最终的融合答案
        4. 评估融合答案的置信度 (0-1)
        
        请以以下 JSON 格式输出：
        {{
            "content": "最终融合答案",
            "highlights": ["模型1的优点", "模型2的优点"],
            "critique": "对各模型的批评",
            "confidence": 置信度
        }}
        """
        
        provider = self.api_manager.select_best_provider(task_type="quality")
        if not provider:
            # 简单策略：返回最长的答案
            best_result = max(results, key=lambda r: len(r["content"]))
            return {
                "content": best_result["content"],
                "provider": best_result["provider"],
                "model": best_result["model"],
                "confidence": 0.7,
                "models_used": [r["provider"] for r in results],
                "time": sum(r["time"] for r in results)
            }
        
        response = self.api_manager.call_api(
            provider,
            [
                {"role": "system", "content": "你是神经符号融合专家，必须以 JSON 格式输出。"},
                {"role": "user", "content": synthesis_prompt}
            ],
            temperature=0.5
        )
        
        if response.get("success"):
            try:
                json_match = re.search(r'\{.*\}', response["content"], re.DOTALL)
                if json_match:
                    synthesis_result = json.loads(json_match.group())
                    return {
                        "content": synthesis_result.get("content", response["content"]),
                        "provider": provider,
                        "model": response["model"],
                        "confidence": synthesis_result.get("confidence", 0.8),
                        "models_used": [r["provider"] for r in results],
                        "time": sum(r["time"] for r in results) + response["time"],
                        "synthesis_details": synthesis_result
                    }
            except:
                pass
        
        # 回退到最长答案
        best_result = max(results, key=lambda r: len(r["content"]))
        return {
            "content": best_result["content"],
            "provider": best_result["provider"],
            "model": best_result["model"],
            "confidence": 0.7,
            "models_used": [r["provider"] for r in results],
            "time": sum(r["time"] for r in results)
        }

# ==================== 进化预测系统 ====================

class EvolutionPredictor:
    """进化预测系统 - 机器学习增强"""
    def __init__(self, cache: AsyncCache):
        self.cache = cache
        self.history = deque(maxlen=1000)
        self.predictions = deque(maxlen=100)
    
    def add_snapshot(self, snapshot: ConsciousnessSnapshot):
        """添加快照"""
        self.history.append(snapshot)
    
    def predict(self, horizon: int = 5, method: str = "auto") -> Dict[str, Any]:
        """预测进化"""
        if len(self.history) < 3:
            return {"error": "历史数据不足，至少需要 3 次测试记录"}
        
        try:
            snapshots = list(self.history)
            dimensions = self.history[0].scores.keys()
            
            predictions = {}
            for dim in dimensions:
                # 提取时间序列
                values = [s.scores[dim] for s in snapshots]
                
                # 多种预测方法
                methods_result = {}
                
                # 1. 线性回归
                lr_pred = self._linear_regression_prediction(values, horizon)
                methods_result["linear_regression"] = lr_pred
                
                # 2. 移动平均
                ma_pred = self._moving_average_prediction(values, horizon)
                methods_result["moving_average"] = ma_pred
                
                # 3. 指数平滑
                es_pred = self._exponential_smoothing_prediction(values, horizon)
                methods_result["exponential_smoothing"] = es_pred
                
                # 4. 自适应方法（根据历史选择最佳）
                if method == "auto":
                    best_method = self._select_best_method(dim, values, methods_result)
                else:
                    best_method = method
                
                predictions[dim] = {
                    "current": values[-1],
                    "predicted": methods_result[best_method][-1],
                    "trajectory": methods_result[best_method],
                    "method_used": best_method,
                    "trend": "上升" if methods_result[best_method][-1] > values[-1] else "下降",
                    "confidence": min(0.95, len(self.history) / 100)
                }
            
            # 整体进化趋势
            total_current = sum(p["current"] for p in predictions.values())
            total_predicted = sum(p["predicted"] for p in predictions.values())
            overall_trend = "快速进化期" if (total_predicted - total_current) > 5 else \
                           "稳步提升中" if (total_predicted - total_current) > 0 else "平稳过渡期"
            
            return {
                "predictions": predictions,
                "overall_trend": overall_trend,
                "total_current": total_current,
                "total_predicted": total_predicted,
                "recommendations": self._generate_recommendations(predictions)
            }
        
        except Exception as e:
            return {"error": f"预测失败: {str(e)}"}
    
    def _linear_regression_prediction(self, values: List[float], horizon: int) -> List[float]:
        """线性回归预测"""
        x = np.arange(len(values))
        z = np.polyfit(x, values, 1)
        p = np.poly1d(z)
        future_x = np.arange(len(values), len(values) + horizon)
        future_values = p(future_x)
        return np.clip(future_values, 0, 14).tolist()
    
    def _moving_average_prediction(self, values: List[float], horizon: int) -> List[float]:
        """移动平均预测"""
        if len(values) < 2:
            return [values[-1]] * horizon
        
        # 计算移动平均斜率
        ma_slope = (values[-1] - values[-2]) if len(values) >= 2 else 0
        predictions = []
        for i in range(horizon):
            next_value = values[-1] + ma_slope * (i + 1)
            predictions.append(np.clip(next_value, 0, 14))
        return predictions
    
    def _exponential_smoothing_prediction(self, values: List[float], horizon: int) -> List[float]:
        """指数平滑预测"""
        alpha = 0.3
        smoothed = values[0]
        for value in values[1:]:
            smoothed = alpha * value + (1 - alpha) * smoothed
        
        # 使用最后的平滑值和趋势
        trend = (values[-1] - values[-2]) if len(values) >= 2 else 0
        predictions = []
        for i in range(horizon):
            next_value = smoothed + trend * (i + 1)
            predictions.append(np.clip(next_value, 0, 14))
        return predictions
    
    def _select_best_method(self, dim: str, values: List[float], 
                            methods: Dict[str, List[float]]) -> str:
        """选择最佳预测方法"""
        # 简单策略：选择方差最小的方法
        variances = {}
        for method, predictions in methods.items():
            variances[method] = np.var(predictions)
        
        return min(variances.keys(), key=lambda k: variances[k])
    
    def _generate_recommendations(self, predictions: Dict) -> List[str]:
        """生成建议"""
        recommendations = []
        
        for dim, pred in predictions.items():
            if pred["predicted"] < pred["current"]:
                recommendations.append(
                    f"{dim}: 呈下降趋势，建议加强针对性训练"
                )
            elif pred["predicted"] - pred["current"] > 1.5:
                recommendations.append(
                    f"{dim}: 进化趋势良好，继续保持"
                )
        
        return recommendations

# ==================== 游戏化系统 ====================

class GamificationSystem:
    """游戏化系统"""
    def __init__(self, config: SystemConfig):
        self.config = config
        self.user_profile = {
            "level": 1,
            "exp": 0,
            "badges": [],
            "streak": 0,
            "last_active": datetime.now()
        }
    
    def add_exp(self, amount: int, reason: str = "") -> Dict[str, Any]:
        """增加经验值"""
        self.user_profile["exp"] += amount
        self.user_profile["last_active"] = datetime.now()
        
        # 检查升级
        old_level = self.user_profile["level"]
        self.user_profile["level"] = self._calculate_level()
        new_level = self.user_profile["level"]
        
        level_up = new_level > old_level
        level_rewards = []
        
        if level_up:
            for level in range(old_level + 1, new_level + 1):
                reward = self._get_level_reward(level)
                if reward:
                    level_rewards.append(reward)
        
        return {
            "exp_gained": amount,
            "total_exp": self.user_profile["exp"],
            "old_level": old_level,
            "new_level": new_level,
            "level_up": level_up,
            "rewards": level_rewards,
            "reason": reason
        }
    
    def _calculate_level(self) -> int:
        """计算等级"""
        exp = self.user_profile["exp"]
        base = self.config.level_config["exp_base"]
        growth = self.config.level_config["exp_growth"]
        max_level = self.config.level_config["max_level"]
        
        # 使用对数增长公式
        level = int(np.log(exp / base + 1) / np.log(growth)) + 1
        return min(level, max_level)
    
    def _get_level_reward(self, level: int) -> Optional[str]:
        """获取等级奖励"""
        rewards = {
            5: "解锁：自定义题目",
            10: "解锁：多模型融合对话",
            20: "解锁：高级预测模式",
            30: "解锁：知识图谱可视化",
            50: "解锁：完整系统权限",
            100: "解锁：意识大师称号"
        }
        return rewards.get(level)
    
    def check_achievements(self, snapshot: ConsciousnessSnapshot, 
                           test_count: int, doc_count: int, 
                           conversation_count: int) -> List[Dict[str, Any]]:
        """检查成就"""
        new_achievements = []
        
        # 检查各种成就条件
        if test_count >= 1 and "first_test" not in [a["id"] for a in self.user_profile["badges"]]:
            new_achievements.append(self._unlock_achievement("first_test"))
        
        if test_count >= 10 and "test_master" not in [a["id"] for a in self.user_profile["badges"]]:
            new_achievements.append(self._unlock_achievement("test_master"))
        
        if snapshot.total_score >= 80 and "perfect_score" not in [a["id"] for a in self.user_profile["badges"]]:
            new_achievements.append(self._unlock_achievement("perfect_score"))
        
        if self.user_profile["level"] >= 10 and "level_10" not in [a["id"] for a in self.user_profile["badges"]]:
            new_achievements.append(self._unlock_achievement("level_10"))
        
        if doc_count >= 10 and "knowledge_collector" not in [a["id"] for a in self.user_profile["badges"]]:
            new_achievements.append(self._unlock_achievement("knowledge_collector"))
        
        if conversation_count >= 100 and "conversation_100" not in [a["id"] for a in self.user_profile["badges"]]:
            new_achievements.append(self._unlock_achievement("conversation_100"))
        
        # 检查全能大师
        all_high = all(score >= 8 for score in snapshot.scores.values())
        if all_high and "all_dimensions" not in [a["id"] for a in self.user_profile["badges"]]:
            new_achievements.append(self._unlock_achievement("all_dimensions"))
        
        return new_achievements
    
    def _unlock_achievement(self, achievement_id: str) -> Dict[str, Any]:
        """解锁成就"""
        achievement_config = self.config.achievements_config[achievement_id]
        achievement = {
            "id": achievement_id,
            "name": achievement_config["name"],
            "desc": achievement_config["desc"],
            "exp": achievement_config["exp"],
            "unlocked_at": datetime.now().isoformat()
        }
        self.user_profile["badges"].append(achievement)
        self.add_exp(achievement_config["exp"], f"成就解锁: {achievement['name']}")
        return achievement

# ==================== 可视化系统 ====================

class AdvancedVisualizer:
    """高级可视化系统"""
    def __init__(self):
        self.colors = {
            "primary": "#6366F1",
            "secondary": "#8B5CF6",
            "accent": "#EC4899",
            "success": "#10B981",
            "warning": "#F59E0B",
            "error": "#EF4444"
        }
    
    def generate_radar_chart(self, scores: Dict[str, float], 
                             history: List[Dict] = None) -> str:
        """生成雷达图"""
        try:
            dimensions = list(scores.keys())
            values = list(scores.values())
            values += values[:1]
            
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, polar=True)
            
            angles = np.linspace(0, 2*np.pi, len(dimensions), endpoint=False).tolist()
            angles += angles[:1]
            
            # 当前状态
            ax.plot(angles, values, 'o-', linewidth=3, 
                   color=self.colors["primary"], 
                   label='当前状态', markersize=8)
            ax.fill(angles, values, alpha=0.3, color=self.colors["primary"])
            
            # 历史轨迹
            if history and len(history) >= 2:
                colors = [self.colors["secondary"], self.colors["accent"], "#60A5FA"]
                for i, hist in enumerate(history[-3:]):
                    hist_values = [hist.get(dim, 0) for dim in dimensions]
                    hist_values += hist_values[:1]
                    alpha = 0.15 + (i / 3) * 0.25
                    ax.plot(angles, hist_values, '-', linewidth=2, 
                           alpha=alpha, color=colors[i % len(colors)],
                           label=f'历史{i+1}')
            
            ax.set_thetagrids(np.degrees(angles[:-1]), dimensions, fontsize=11, fontweight='bold')
            ax.set_ylim(0, 14)
            ax.set_title('意识强度雷达图', fontsize=18, pad=25, fontweight='bold')
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
            ax.grid(True, alpha=0.3)
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=200, facecolor='#0F0F1A')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close()
            
            return f"data:image/png;base64,{img_base64}"
        
        except Exception as e:
            print(f"Radar chart error: {e}")
            return None
    
    def generate_evolution_chart(self, history: List[Dict]) -> str:
        """生成进化趋势图"""
        try:
            if len(history) < 2:
                return None
            
            dimensions = list(history[0].get('category_scores', {}).keys())
            timestamps = [datetime.fromisoformat(h['timestamp']) for h in history]
            
            fig, axes = plt.subplots(3, 4, figsize=(16, 12))
            fig.suptitle('意识进化趋势', fontsize=18, fontweight='bold')
            
            for idx, dim in enumerate(dimensions):
                row = idx // 4
                col = idx % 4
                ax = axes[row, col]
                
                values = [h.get('category_scores', {}).get(dim, 0) for h in history]
                ax.plot(timestamps, values, marker='o', linewidth=2, 
                       color=self.colors["primary"], markersize=6)
                ax.fill_between(timestamps, values, alpha=0.2, color=self.colors["primary"])
                ax.set_title(dim, fontsize=10, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.set_ylim(0, 14)
                
                # 旋转 x 轴标签
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=150, facecolor='#0F0F1A')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close()
            
            return f"data:image/png;base64,{img_base64}"
        
        except Exception as e:
            print(f"Evolution chart error: {e}")
            return None

# ==================== 数据管理 ====================

class DataManager:
    """数据管理器"""
    def __init__(self):
        self.test_results = []
        self.conversation_history = []
        self.user_data = {}
    
    def save_test_result(self, result: TestResult):
        """保存测试结果"""
        self.test_results.append({
            "timestamp": result.snapshot.timestamp.isoformat(),
            "scores": result.snapshot.scores,
            "total_score": result.snapshot.total_score,
            "level": result.snapshot.level.value,
            "question_count": result.question_count,
            "accuracy": result.accuracy
        })
    
    def save_conversation(self, message: ConversationMessage):
        """保存对话"""
        self.conversation_history.append({
            "role": message.role,
            "content": message.content,
            "timestamp": message.timestamp.isoformat(),
            "confidence": message.confidence
        })
    
    def export_data(self) -> Dict[str, Any]:
        """导出数据"""
        return {
            "export_time": datetime.now().isoformat(),
            "test_results": self.test_results,
            "conversation_history": self.conversation_history[-50:],  # 最近 50 条
            "user_data": self.user_data
        }
    
    def import_data(self, data: Dict[str, Any]):
        """导入数据"""
        if "test_results" in data:
            self.test_results.extend(data["test_results"])
        if "conversation_history" in data:
            self.conversation_history.extend(data["conversation_history"])
        if "user_data" in data:
            self.user_data.update(data["user_data"])

# ==================== 主系统 ====================

class MultiverseSystem:
    """多维轮回破解系统主类"""
    def __init__(self):
        self.config = SystemConfig()
        self.api_keys = self._load_api_keys()
        self.cache = AsyncCache()
        
        # 初始化子系统
        self.api_manager = APIManager(self.api_keys, self.config)
        self.knowledge_base = AdvancedKnowledgeBase(self.cache)
        self.question_generator = AIQuestionGenerator(self.api_manager, self.knowledge_base)
        self.grading_engine = AIGradingEngine(self.api_manager, self.knowledge_base)
        self.dialogue_system = MultiModelDialogueSystem(self.api_manager, self.cache)
        self.predictor = EvolutionPredictor(self.cache)
        self.gamification = GamificationSystem(self.config)
        self.visualizer = AdvancedVisualizer()
        self.data_manager = DataManager()
    
    def _load_api_keys(self) -> Dict[str, str]:
        """加载 API 密钥"""
        keys = {}
        
        # 从 Streamlit Secrets 加载
        if hasattr(st, 'secrets'):
            keys['groq'] = st.secrets.get('GROQ_API_KEY', '')
            keys['openai'] = st.secrets.get('OPENAI_API_KEY', '')
            keys['anthropic'] = st.secrets.get('ANTHROPIC_API_KEY', '')
            keys['cohere'] = st.secrets.get('COHERE_API_KEY', '')
        
        # 从环境变量加载
        if not keys.get('groq'):
            keys['groq'] = os.environ.get('GROQ_API_KEY', '')
        if not keys.get('openai'):
            keys['openai'] = os.environ.get('OPENAI_API_KEY', '')
        if not keys.get('anthropic'):
            keys['anthropic'] = os.environ.get('ANTHROPIC_API_KEY', '')
        if not keys.get('cohere'):
            keys['cohere'] = os.environ.get('COHERE_API_KEY', '')
        
        return keys
    
    def get_api_status(self) -> Dict[str, Any]:
        """获取 API 状态"""
        status = {}
        for provider, config in self.config.api_providers.items():
            has_key = bool(self.api_keys.get(provider))
            health = self.api_manager.health_status[provider]
            stats = self.api_manager.usage_stats[provider]
            
            status[provider] = {
                "name": config["name"],
                "configured": has_key,
                "available": health["available"],
                "free": config["free"],
                "requests": stats["requests"],
                "success_rate": stats["success"] / stats["requests"] if stats["requests"] > 0 else 0,
                "avg_time": f"{stats['avg_time']:.2f}s"
            }
        
        return status

# ==================== Streamlit 界面 ====================

def initialize_session_state():
    """初始化会话状态"""
    if 'system' not in st.session_state:
        st.session_state.system = MultiverseSystem()
    
    if 'test_in_progress' not in st.session_state:
        st.session_state.test_in_progress = False
    
    if 'current_questions' not in st.session_state:
        st.session_state.current_questions = []
    
    if 'current_question_index' not in st.session_state:
        st.session_state.current_question_index = 0
    
    if 'test_answers' not in st.session_state:
        st.session_state.test_answers = []

def show_custom_css():
    """显示自定义 CSS"""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    body {
        background: linear-gradient(135deg, #0F0F1A 0%, #1A1A2E 50%, #16213E 100%);
        color: #F1F5F9;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background: transparent;
    }
    
    .title-gradient {
        background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 50%, #EC4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 20px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
    }
    
    .stat-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    
    .progress-container {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        overflow: hidden;
    }
    .progress-bar {
        background: linear-gradient(90deg, #6366F1 0%, #8B5CF6 100%);
        height: 10px;
        transition: width 0.5s ease;
    }
    
    .badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        margin: 4px;
        background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%);
        box-shadow: 0 2px 8px rgba(99, 102, 241, 0.3);
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.4);
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    .animate-pulse {
        animation: pulse 2s ease-in-out infinite;
    }
    
    @media (max-width: 768px) {
        h1 { font-size: 1.8em !important; }
        h2 { font-size: 1.5em !important; }
        .glass-card { padding: 16px; }
    }
    </style>
    """, unsafe_allow_html=True)

def show_header():
    """显示标题"""
    st.markdown("""
    <div style="text-align: center; padding: 30px 0;">
        <h1 class="title-gradient" style="font-size: 2.5em; margin: 0; font-weight: 700;">
            多维轮回破解系统
        </h1>
        <p style="color: #94A3B8; font-size: 1.2em; margin-top: 10px; font-weight: 300;">
            渊开发 v3.0 Lite - 终极进化版
        </p>
        <div style="margin-top: 15px; flex-wrap: wrap; display: flex; justify-content: center; gap: 8px;">
            <span class="badge">AI 智能出题</span>
            <span class="badge">AI 深度评分</span>
            <span class="badge">多模型融合</span>
            <span class="badge">机器学习预测</span>
            <span class="badge">向量检索</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def show_sidebar():
    """显示侧边栏"""
    system = st.session_state.system
    profile = system.gamification.user_profile
    
    st.markdown("""
    <div class="glass-card" style="margin-bottom: 20px;">
        <h3 style="color: #6366F1; margin: 0 0 15px 0;">🧠 意识进化状态</h3>
        <div style="text-align: center; margin-bottom: 15px;">
            <div style="font-size: 3em; font-weight: 700; color: #8B5CF6;">
                Lv.{profile['level']}
            </div>
            <div style="color: #94A3B8;">当前等级</div>
        </div>
        <div class="progress-container" style="margin-bottom: 10px;">
            <div class="progress-bar" style="width: 50%;"></div>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 0.9em; color: #64748B;">
            <span>经验值: {exp}</span>
            <span>下一级: {next_exp}</span>
        </div>
    </div>
    """.format(
        exp=profile['exp'],
        next_exp=int(100 * (profile['level'] ** 1.2))
    ), unsafe_allow_html=True)
    
    # 显示徽章
    if profile['badges']:
        st.markdown("### 🏆 成就徽章")
        for badge in profile['badges'][-5:]:  # 最近 5 个
            st.markdown(f"<span class='badge'>{badge['name']}</span>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 功能导航
    page = st.selectbox(
        "选择功能",
        ["🧠 意识测试", "🔮 智能对话", "📊 进化预测", "📚 知识库", "⚙️ 系统设置"],
        label_visibility="collapsed"
    )
    
    return page

def consciousness_test_page():
    """意识测试页面"""
    st.markdown("## 意识强度深度评估")
    
    system = st.session_state.system
    
    if not st.session_state.test_in_progress:
        # 测试配置
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            <div class="glass-card">
                <h3 style="color: #6366F1;">AI 智能出题系统</h3>
                <p style="color: #94A3B8; line-height: 1.6;">
                    系统将根据你的知识库和 AI 能力，智能生成个性化测试题目。
                    题目涵盖 12 个意识维度，真实评估你的认知能力。
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # 测试配置
            test_mode = st.radio(
                "测试模式",
                ["快速测试 (3题)", "标准测试 (6题)", "深度测试 (10题)"],
                horizontal=True
            )
            
            question_count = 3 if "快速" in test_mode else 6 if "标准" in test_mode else 10
            
            use_knowledge = st.checkbox("基于知识库出题", value=True, 
                                       help="如果你的知识库有内容，题目会更相关")
            
            if st.button("🚀 启动测试", type="primary", use_container_width=True):
                with st.spinner("AI 正在智能出题..."):
                    # 选择测试维度
                    dimensions = random.sample(system.config.consciousness_dimensions, 
                                            min(question_count, len(system.config.consciousness_dimensions)))
                    
                    # 生成题目
                    questions = system.question_generator.generate_batch_questions(
                        dimensions=dimensions,
                        count_per_dimension=1
                    )
                    
                    if len(questions) < question_count:
                        st.error("题目生成失败，请检查 API 配置")
                        return
                    
                    st.session_state.current_questions = questions[:question_count]
                    st.session_state.current_question_index = 0
                    st.session_state.test_answers = []
                    st.session_state.test_in_progress = True
                    st.success(f"成功生成 {len(st.session_state.current_questions)} 道题目！")
                    st.rerun()
        
        with col2:
            st.markdown("### 📊 统计信息")
            
            kb_stats = system.knowledge_base.get_statistics()
            st.markdown(f"""
            <div class="stat-card" style="margin-bottom: 10px;">
                <div style="font-size: 2em; font-weight: 700; color: #6366F1;">
                    {kb_stats['total_documents']}
                </div>
                <div style="color: #94A3B8;">知识库文档</div>
            </div>
            <div class="stat-card">
                <div style="font-size: 2em; font-weight: 700; color: #8B5CF6;">
                    {len(system.data_manager.test_results)}
                </div>
                <div style="color: #94A3B8;">历史测试</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        # 进行测试
        current_question = st.session_state.current_questions[st.session_state.current_question_index]
        progress = (st.session_state.current_question_index + 1) / len(st.session_state.current_questions)
        
        st.markdown(f"""
        <div class="glass-card" style="margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                <h4 style="color: #8B5CF6; margin: 0;">{current_question.dimension}</h4>
                <span style="color: #94A3B8; font-size: 0.9em;">
                    问题 {st.session_state.current_question_index + 1} / {len(st.session_state.current_questions)}
                </span>
            </div>
            <div class="progress-container" style="margin-bottom: 15px;">
                <div class="progress-bar" style="width: {progress * 100}%;"></div>
            </div>
            <p style="font-size: 1.1em; color: #F1F5F9; line-height: 1.8;">
                {current_question.content}
            </p>
            {f"<p style='color: #64748B; font-size: 0.85em; margin-top: 10px;'>参考: {current_question.reference}</p>" if current_question.reference else ""}
        </div>
        """, unsafe_allow_html=True)
        
        # 答题区域
        answer = st.text_area("✍️ 您的回答", height=150, 
                             placeholder="请输入您的回答，越详细越好...",
                             key=f"answer_{st.session_state.current_question_index}")
        
        col_next, col_skip = st.columns([1, 1])
        
        with col_next:
            if st.button("下一题 ➡️", use_container_width=True):
                if answer:
                    st.session_state.test_answers.append({
                        "question": current_question,
                        "answer": answer
                    })
                    st.session_state.current_question_index += 1
                    
                    # 检查是否完成
                    if st.session_state.current_question_index >= len(st.session_state.current_questions):
                        # 测试完成，进行评分
                        complete_test(system)
                    else:
                        st.rerun()
                else:
                    st.warning("请先回答问题")
        
        with col_skip:
            if st.button("跳过", use_container_width=True):
                st.session_state.test_answers.append({
                    "question": current_question,
                    "answer": "",
                    "skipped": True
                })
                st.session_state.current_question_index += 1
                
                if st.session_state.current_question_index >= len(st.session_state.current_questions):
                    complete_test(system)
                else:
                    st.rerun()

def complete_test(system):
    """完成测试"""
    with st.spinner("AI 正在深度分析您的答案..."):
        scores = {}
        grading_results = {}
        
        for answer_data in st.session_state.test_answers:
            if "skipped" not in answer_data:
                question = answer_data["question"]
                answer = answer_data["answer"]
                
                # AI 评分
                grading = system.grading_engine.grade_answer(question, answer)
                grading_results[question.dimension] = grading
                
                # 计算维度分数
                scores[question.dimension] = grading.get("total_score", 7)
        
        # 填充未测试的维度
        for dim in system.config.consciousness_dimensions:
            if dim not in scores:
                scores[dim] = random.uniform(4, 7)
        
        total_score = sum(scores.values())
        
        # 创建快照
        snapshot = ConsciousnessSnapshot(
            timestamp=datetime.now(),
            scores=scores,
            level=ConsciousnessLevel(min(7, int(total_score / 15) + 1)),
            total_score=total_score,
            test_answers=st.session_state.test_answers
        )
        
        # 保存到历史
        system.predictor.add_snapshot(snapshot)
        
        # 创建测试结果
        test_result = TestResult(
            snapshot=snapshot,
            question_count=len(st.session_state.current_questions),
            correct_count=len(st.session_state.test_answers),
            accuracy=1.0,
            time_spent=0,
            confidence_scores={dim: grading_results.get(dim, {}).get("confidence", 0.8) 
                              for dim in scores.keys()}
        )
        
        # 保存数据
        system.data_manager.save_test_result(test_result)
        
        # 经验值和成就
        exp_gained = 50 + int(total_score * 2)
        exp_result = system.gamification.add_exp(exp_gained, "完成意识测试")
        
        new_achievements = system.gamification.check_achievements(
            snapshot,
            len(system.data_manager.test_results),
            system.knowledge_base.get_statistics()["total_documents"],
            len(system.data_manager.conversation_history)
        )
        
        # 清理测试状态
        st.session_state.test_in_progress = False
        st.session_state.current_questions = []
        st.session_state.current_question_index = 0
        st.session_state.test_answers = []
        
        # 显示结果
        st.session_state.show_test_result = {
            "snapshot": snapshot,
            "grading_results": grading_results,
            "exp_result": exp_result,
            "achievements": new_achievements
        }
    
    st.rerun()

def show_test_result():
    """显示测试结果"""
    if 'show_test_result' not in st.session_state:
        return
    
    result_data = st.session_state.show_test_result
    snapshot = result_data["snapshot"]
    grading_results = result_data["grading_results"]
    exp_result = result_data["exp_result"]
    achievements = result_data["achievements"]
    
    system = st.session_state.system
    
    st.success("🎉 测试完成！AI 已深度分析您的答案")
    
    # 总体得分
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div style="font-size: 2.5em; font-weight: 700; color: #8B5CF6;">
                {snapshot.total_score:.1f}
            </div>
            <div style="color: #94A3B8;">总分</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div style="font-size: 2.5em; font-weight: 700; color: #10B981;">
                +{exp_result['exp_gained']}
            </div>
            <div style="color: #94A3B8;">经验值</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        if exp_result["level_up"]:
            st.markdown(f"""
            <div class="stat-card" style="border-color: #EC4899;">
                <div style="font-size: 2.5em; font-weight: 700; color: #EC4899;">
                    🎖️
                </div>
                <div style="color: #94A3B8;">等级提升！</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="stat-card">
                <div style="font-size: 2.5em; font-weight: 700; color: #6366F1;">
                    Lv.{exp_result['new_level']}
                </div>
                <div style="color: #94A3B8;">当前等级</div>
            </div>
            """, unsafe_allow_html=True)
    
    # 详细评分
    st.markdown("### 📊 各维度详细评分")
    
    for dim, score in snapshot.scores.items():
        grading = grading_results.get(dim, {})
        confidence = grading.get("confidence", 0.8)
        
        with st.expander(f"**{dim}**: {score:.2f} 分 (置信度: {confidence:.0%})", expanded=False):
            if grading:
                st.markdown(f"**反馈**: {grading.get('feedback', '无反馈')}")
                
                if grading.get('strengths'):
                    st.markdown("**优势**:")
                    for strength in grading['strengths']:
                        st.markdown(f"- {strength}")
                
                if grading.get('weaknesses'):
                    st.markdown("**改进建议**:")
                    for weakness in grading['weaknesses']:
                        st.markdown(f"- {weakness}")
    
    # 新成就
    if achievements:
        st.markdown("### 🏆 解锁新成就")
        for achievement in achievements:
            st.success(f"🎊 {achievement['name']}: {achievement['desc']} (+{achievement['exp']} 经验)")
    
    # 雷达图
    st.markdown("### 🎯 能力雷达图")
    
    history = [
        {
            "category_scores": r["scores"]
        }
        for r in system.data_manager.test_results[-5:]
    ]
    
    radar_chart = system.visualizer.generate_radar_chart(snapshot.scores, history)
    if radar_chart:
        st.image(radar_chart, use_column_width=True)
    
    if st.button("完成", use_container_width=True):
        del st.session_state.show_test_result
        st.rerun()

def dialogue_page():
    """对话页面"""
    st.markdown("## 🔮 多模型融合对话")
    
    system = st.session_state.system
    
    # 显示对话历史
    chat_container = st.container()
    
    with chat_container:
        if not system.dialogue_system.conversation_history and not system.data_manager.conversation_history:
            st.markdown("""
            <div style="text-align: center; padding: 40px; color: #94A3B8;">
                <h3>🌌 开始您的深度对话</h3>
                <p>系统将运用多模型融合推理为您提供深度洞察</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # 显示历史对话
            for msg in system.data_manager.conversation_history[-20:]:
                if msg["role"] == "user":
                    st.markdown(f"""
                    <div class="glass-card" style="background: rgba(99, 102, 241, 0.1);">
                        <strong>您:</strong> {msg['content']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="glass-card">
                        <strong>系统:</strong> {msg['content']}
                    </div>
                    """, unsafe_allow_html=True)
    
    # 输入区域
    col_input, col_clear = st.columns([5, 1])
    
    with col_input:
        user_input = st.text_input("💭 输入您的问题...", key="chat_input", 
                                   placeholder="请输入您的问题，越详细越好...")
    
    with col_clear:
        if st.button("🗑️ 清空"):
            system.data_manager.conversation_history = []
            system.dialogue_system.conversation_history.clear()
            st.rerun()
    
    # 配置选项
    with st.expander("⚙️ 对话配置", expanded=False):
        use_ensemble = st.checkbox("多模型融合", value=True, 
                                  help="同时调用多个 API 并融合结果")
        complexity = st.slider("推理复杂度", 1, 10, 7)
    
    if st.button("🚀 发送", type="primary", use_container_width=True):
        if user_input:
            # 保存用户消息
            system.data_manager.save_conversation(
                ConversationMessage(
                    role="user",
                    content=user_input,
                    timestamp=datetime.now()
                )
            )
            
            with st.spinner("多模型融合推理中..."):
                result = system.dialogue_system.dialogue(
                    user_input=user_input,
                    context="",
                    complexity=complexity,
                    use_ensemble=use_ensemble
                )
            
            if result.get("content"):
                # 保存系统回复
                system.data_manager.save_conversation(
                    ConversationMessage(
                        role="assistant",
                        content=result["content"],
                        timestamp=datetime.now(),
                        confidence=result.get("confidence", 0.8),
                        models_used=result.get("models_used", [])
                    )
                )
                
                # 增加经验值
                exp_gained = 5 + len(result.get("models_used", [])) * 2
                system.gamification.add_exp(exp_gained, "完成对话")
                
                st.rerun()
            else:
                st.error(f"对话失败: {result.get('error', '未知错误')}")

def prediction_page():
    """预测页面"""
    st.markdown("## 📊 意识进化预测")
    
    system = st.session_state.system
    
    if len(system.data_manager.test_results) < 3:
        st.info(f"""
        ⚠️ **需要更多历史数据**
        
        至少需要完成 **3 次意识测试**才能启用进化预测功能。
        
        当前已完成: {len(system.data_manager.test_results)} / 3 次
        """)
        return
    
    # 执行预测
    with st.spinner("AI 正在分析历史数据，预测进化轨迹..."):
        prediction = system.predictor.predict(horizon=5)
    
    if prediction.get("error"):
        st.error(f"预测失败: {prediction['error']}")
        return
    
    # 显示预测结果
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown(f"""
        <div class="glass-card">
            <h3 style="color: #6366F1;">整体进化趋势</h3>
            <div style="font-size: 2em; font-weight: 700; color: #8B5CF6; margin: 20px 0;">
                {prediction['overall_trend']}
            </div>
            <div style="display: flex; justify-content: space-between;">
                <div>
                    <div style="color: #94A3B8; font-size: 0.9em;">当前总分</div>
                    <div style="font-size: 1.5em; font-weight: 600; color: #F1F5F9;">
                        {prediction['total_current']:.1f}
                    </div>
                </div>
                <div>
                    <div style="color: #94A3B8; font-size: 0.9em;">预测总分</div>
                    <div style="font-size: 1.5em; font-weight: 600; color: #10B981;">
                        {prediction['total_predicted']:.1f}
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if prediction.get('recommendations'):
            st.markdown("""
            <div class="glass-card">
                <h3 style="color: #6366F1;">进化建议</h3>
            </div>
            """, unsafe_allow_html=True)
            
            for rec in prediction['recommendations']:
                st.markdown(f"- {rec}")
    
    # 详细预测
    st.markdown("### 📈 各维度预测详情")
    
    pred_data = []
    for dim, pred in prediction['predictions'].items():
        pred_data.append({
            "维度": dim,
            "当前分数": f"{pred['current']:.2f}",
            "预测分数": f"{pred['predicted']:.2f}",
            "变化": f"{pred['predicted'] - pred['current']:+.2f}",
            "趋势": pred['trend'],
            "置信度": f"{pred['confidence']:.0%}",
            "预测方法": pred['method_used']
        })
    
    df = pd.DataFrame(pred_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # 进化趋势图
    st.markdown("### 📊 历史进化趋势")
    
    history = [
        {
            "timestamp": r["timestamp"],
            "category_scores": r["scores"]
        }
        for r in system.data_manager.test_results
    ]
    
    evolution_chart = system.visualizer.generate_evolution_chart(history)
    if evolution_chart:
        st.image(evolution_chart, use_column_width=True)

def knowledge_base_page():
    """知识库页面"""
    st.markdown("## 📚 高级知识库")
    
    system = st.session_state.system
    
    tab1, tab2, tab3 = st.tabs(["📤 上传文档", "🔍 智能检索", "📊 统计信息"])
    
    with tab1:
        st.markdown("### 上传您的知识文档")
        
        uploaded_file = st.file_uploader(
            "选择文件",
            type=['txt', 'md', 'pdf', 'docx'],
            help="支持上传文本、Markdown、PDF 和 Word 文档"
        )
        
        if uploaded_file:
            try:
                # 处理文件
                if uploaded_file.type == "application/pdf":
                    import PyPDF2
                    reader = PyPDF2.PdfReader(uploaded_file)
                    content = ""
                    for page in reader.pages:
                        content += page.extract_text()
                elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    from docx import Document
                    doc = Document(uploaded_file)
                    content = "\n".join([para.text for para in doc.paragraphs])
                else:
                    content = uploaded_file.read().decode('utf-8')
                
                if content.strip():
                    doc_id = f"{uploaded_file.name}_{int(time.time())}"
                    system.knowledge_base.add_document(
                        doc_id=doc_id,
                        content=content,
                        metadata={
                            "filename": uploaded_file.name,
                            "size": uploaded_file.size,
                            "type": uploaded_file.type,
                            "uploaded_at": datetime.now().isoformat()
                        }
                    )
                    
                    st.success(f"✅ 文档已成功添加到知识库")
                    
                    # 增加经验值
                    exp_gained = 15
                    exp_result = system.gamification.add_exp(exp_gained, "上传知识文档")
                    
                    st.info(f"🎁 获得 {exp_gained} 经验值")
                else:
                    st.warning("文档内容为空，请检查文件")
            
            except Exception as e:
                st.error(f"文档处理失败: {str(e)}")
    
    with tab2:
        st.markdown("### 智能语义检索")
        
        col_search = st.columns([3, 1])
        
        with col_search[0]:
            search_query = st.text_input("🔎 输入搜索查询", key="kb_search",
                                       placeholder="请输入搜索关键词...")
        
        with col_search[1]:
            top_k = st.selectbox("结果数量", [3, 5, 10], index=1)
        
        if st.button("🔍 搜索") and search_query:
            with st.spinner("正在检索知识库..."):
                results = system.knowledge_base.search(search_query, top_k=top_k)
            
            if results:
                for i, result in enumerate(results, 1):
                    st.markdown(f"""
                    <div class="glass-card">
                        <h4 style="color: #6366F1;">结果 {i} - {result['metadata'].get('filename', '未知文件')}</h4>
                        <div style="display: flex; gap: 15px; margin: 10px 0;">
                            <span style="color: #94A3B8; font-size: 0.85em;">
                                相关度: <strong style="color: #10B981;">{result['score']:.2%}</strong>
                            </span>
                            <span style="color: #94A3B8; font-size: 0.85em;">
                                文档大小: {result['metadata'].get('size', 0) / 1024:.1f} KB
                            </span>
                        </div>
                        <p style="color: #F1F5F9; line-height: 1.6;">
                            {result['content']}...
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("未找到相关文档，请尝试其他关键词")
    
    with tab3:
        st.markdown("### 知识库统计")
        
        stats = system.knowledge_base.get_statistics()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("文档总数", stats["total_documents"])
        with col2:
            st.metric("总词数", f"{stats['total_tokens']:,}")
        with col3:
            st.metric("总字符数", f"{stats['total_length']:,}")
        with col4:
            st.metric("平均长度", f"{stats['avg_document_length']:.0f} 字符")
        
        # 系统性能
        st.markdown("---")
        st.markdown("### ⚡ 系统性能")
        
        cache_stats = system.cache.get_stats()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("缓存命中率", f"{cache_stats['hit_rate']:.2%}")
        with col2:
            st.metric("缓存大小", f"{cache_stats['size']}/{cache_stats['max_size']}")
        with col3:
            st.metric("总请求", cache_stats['hits'] + cache_stats['misses'])

def settings_page():
    """设置页面"""
    st.markdown("## ⚙️ 系统设置")
    
    system = st.session_state.system
    
    # API 状态
    st.markdown("### 🔑 API 状态")
    
    api_status = system.get_api_status()
    
    for provider, status in api_status.items():
        status_icon = "✅" if status["configured"] else "❌"
        available_icon = "🟢" if status["available"] else "🔴"
        
        st.markdown(f"""
        <div class="glass-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h4 style="margin: 0; color: #6366F1;">{status_icon} {status['name']}</h4>
                    <div style="color: #94A3B8; font-size: 0.9em; margin-top: 5px;">
                        {available_icon} {'可用' if status['available'] else '不可用'} | 
                        {'免费' if status['free'] else '付费'}
                    </div>
                </div>
                <div style="text-align: right;">
                    <div style="color: #94A3B8; font-size: 0.85em;">请求数: {status['requests']}</div>
                    <div style="color: #94A3B8; font-size: 0.85em;">成功率: {status['success_rate']:.0%}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.info("""
    **如何配置 API 密钥**
    
    1. 访问 Streamlit Cloud 应用管理页面
    2. 进入 "Settings" → "Secrets"
    3. 添加以下密钥（至少一个）：
       - `GROQ_API_KEY`: Groq API 密钥（推荐，免费）
       - `OPENAI_API_KEY`: OpenAI API 密钥
       - `ANTHROPIC_API_KEY`: Anthropic API 密钥
       - `COHERE_API_KEY`: Cohere API 密钥
    
    **获取免费 API 密钥**: https://console.groq.com/keys
    """)
    
    st.markdown("---")
    
    # 数据管理
    st.markdown("### 💾 数据管理")
    
    col_export, col_clear = st.columns(2)
    
    with col_export:
        if st.button("📥 导出数据", use_container_width=True):
            data = system.data_manager.export_data()
            data_json = json.dumps(data, ensure_ascii=False, indent=2)
            
            st.download_button(
                label="💾 下载数据文件",
                data=data_json,
                file_name=f"multiverse_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col_clear:
        if st.button("🗑️ 清空所有数据", use_container_width=True):
            if st.confirm("⚠️ 确定要清空所有数据吗？此操作不可恢复！"):
                system.data_manager = DataManager()
                system.gamification.user_profile = {
                    "level": 1,
                    "exp": 0,
                    "badges": [],
                    "streak": 0,
                    "last_active": datetime.now()
                }
                system.cache.clear()
                st.success("所有数据已清空")
                st.rerun()
    
    # 关于
    st.markdown("---")
    st.markdown("""
    <div class="glass-card">
        <h3 style="color: #6366F1;">关于系统</h3>
        <div style="color: #94A3B8; line-height: 1.8;">
            <p><strong>版本:</strong> v3.0 Lite - 终极进化版</p>
            <p><strong>开发者:</strong> 渊开发</p>
            <p><strong>核心特性:</strong></p>
            <ul style="margin: 10px 0; padding-left: 20px;">
                <li>AI 智能出题引擎</li>
                <li>AI 深度评分引擎</li>
                <li>多模型融合对话</li>
                <li>机器学习预测</li>
                <li>向量检索知识库</li>
                <li>深度游戏化系统</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

def main():
    """主函数"""
    # 初始化
    initialize_session_state()
    show_custom_css()
    
    # 标题
    show_header()
    
    # 侧边栏
    page = show_sidebar()
    
    # 检查 API 配置
    system = st.session_state.system
    available_apis = system.api_manager.get_available_providers()
    
    if not available_apis:
        st.warning("""
        ⚠️ **未配置 API 密钥**
        
        请在 Streamlit Cloud 的 "Settings" → "Secrets" 中配置至少一个 API 密钥。
        
        **推荐使用免费的 Groq API**:
        1. 访问: https://console.groq.com/
        2. 注册并获取 API 密钥
        3. 在 Streamlit Cloud Secrets 中添加 `GROQ_API_KEY`
        """)
    
    # 显示测试结果
    if 'show_test_result' in st.session_state:
        show_test_result()
        return
    
    # 页面路由
    if page == "🧠 意识测试":
        consciousness_test_page()
    elif page == "🔮 智能对话":
        dialogue_page()
    elif page == "📊 进化预测":
        prediction_page()
    elif page == "📚 知识库":
        knowledge_base_page()
    elif page == "⚙️ 系统设置":
        settings_page()
    
    # 页脚
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; padding: 20px; color: #64748B;">
        <p>多维轮回破解系统 v3.0 Lite - 终极进化版</p>
        <p style="font-size: 0.85em; margin-top: 5px;">
            渊开发 © 2024 | AI 智能出题 | AI 深度评分 | 多模型融合 | 机器学习预测 | 向量检索
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
