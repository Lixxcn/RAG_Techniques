#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG调试工具集

这个模块提供了一套完整的RAG系统调试工具，帮助开发者深入了解RAG流水线中每个环节的工作情况。

主要功能：
1. 详细的流水线调试
2. 检索质量分析
3. 性能监控
4. 结果对比
5. 可视化展示
"""

import time
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import pandas as pd

@dataclass
class RAGDebugResult:
    """RAG调试结果数据类"""
    question: str
    timestamp: str
    retrieval_time: float
    llm_time: float
    total_time: float
    retrieved_docs: List[Any]
    context: str
    final_answer: str
    retrieval_scores: List[float]
    embedding_vector: List[float]
    prompt_tokens: int
    response_tokens: int
    
class RAGDebugger:
    """RAG系统调试器"""
    
    def __init__(self, vector_store, embeddings, llm, prompt_template):
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.llm = llm
        self.prompt_template = prompt_template
        self.debug_history = []
        
    def comprehensive_debug(self, question: str, k: int = 4, verbose: bool = True) -> RAGDebugResult:
        """
        全面的RAG调试分析
        
        Args:
            question: 用户问题
            k: 检索文档数量
            verbose: 是否显示详细信息
            
        Returns:
            RAGDebugResult: 包含所有调试信息的结果对象
        """
        start_time = time.time()
        timestamp = datetime.now().isoformat()
        
        if verbose:
            print("\n" + "="*80)
            print(f"🔍 RAG 全面调试分析 - {timestamp}")
            print("="*80)
            print(f"📝 问题: {question}")
            print(f"🔢 检索数量: {k}")
        
        # 1. 问题向量化
        if verbose:
            print("\n🔢 步骤1: 问题向量化")
        embedding_start = time.time()
        try:
            question_embedding = self.embeddings.embed_query(question)
            embedding_time = time.time() - embedding_start
            if verbose:
                print(f"  ✅ 向量维度: {len(question_embedding)}")
                print(f"  ⏱️ 向量化耗时: {embedding_time:.3f}秒")
                print(f"  📊 向量范围: [{min(question_embedding):.4f}, {max(question_embedding):.4f}]")
        except Exception as e:
            if verbose:
                print(f"  ❌ 向量化失败: {e}")
            question_embedding = []
            embedding_time = 0
        
        # 2. 文档检索
        if verbose:
            print("\n🔍 步骤2: 文档检索")
        retrieval_start = time.time()
        try:
            # 获取带分数的检索结果
            docs_with_scores = self.vector_store.similarity_search_with_score(question, k=k)
            retrieved_docs = [doc for doc, score in docs_with_scores]
            retrieval_scores = [score for doc, score in docs_with_scores]
            retrieval_time = time.time() - retrieval_start
            
            if verbose:
                print(f"  ✅ 检索到 {len(retrieved_docs)} 个文档")
                print(f"  ⏱️ 检索耗时: {retrieval_time:.3f}秒")
                print(f"  📊 相似度分数范围: [{min(retrieval_scores):.4f}, {max(retrieval_scores):.4f}]")
                
                for i, (doc, score) in enumerate(docs_with_scores):
                    print(f"\n    📄 文档 {i+1} (分数: {score:.4f}):")
                    print(f"      内容长度: {len(doc.page_content)} 字符")
                    print(f"      内容预览: {doc.page_content[:100]}...")
                    if hasattr(doc, 'metadata') and doc.metadata:
                        print(f"      元数据: {doc.metadata}")
                        
        except Exception as e:
            if verbose:
                print(f"  ❌ 检索失败: {e}")
            retrieved_docs = []
            retrieval_scores = []
            retrieval_time = 0
        
        # 3. 上下文构建
        if verbose:
            print("\n📋 步骤3: 上下文构建")
        try:
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            if verbose:
                print(f"  ✅ 上下文总长度: {len(context)} 字符")
                print(f"  📝 上下文预览: {context[:200]}...")
        except Exception as e:
            if verbose:
                print(f"  ❌ 上下文构建失败: {e}")
            context = ""
        
        # 4. 提示构建
        if verbose:
            print("\n💬 步骤4: 提示构建")
        try:
            final_prompt = self.prompt_template.format_messages(
                context=context,
                input=question
            )
            prompt_content = "\n".join([msg.content for msg in final_prompt])
            prompt_tokens = len(prompt_content.split())  # 简单的token估算
            
            if verbose:
                print(f"  ✅ 提示消息数量: {len(final_prompt)}")
                print(f"  📊 估算token数: {prompt_tokens}")
                print(f"  📝 提示预览: {prompt_content[:200]}...")
        except Exception as e:
            if verbose:
                print(f"  ❌ 提示构建失败: {e}")
            final_prompt = []
            prompt_tokens = 0
        
        # 5. LLM生成
        if verbose:
            print("\n🤖 步骤5: LLM生成")
        llm_start = time.time()
        try:
            response = self.llm.invoke(final_prompt)
            llm_time = time.time() - llm_start
            
            if hasattr(response, 'content'):
                final_answer = response.content
            else:
                final_answer = str(response)
                
            response_tokens = len(final_answer.split())  # 简单的token估算
            
            if verbose:
                print(f"  ✅ 生成成功")
                print(f"  ⏱️ 生成耗时: {llm_time:.3f}秒")
                print(f"  📊 响应token数: {response_tokens}")
                print(f"  📝 答案: {final_answer}")
                
        except Exception as e:
            if verbose:
                print(f"  ❌ LLM生成失败: {e}")
            final_answer = "生成失败"
            llm_time = 0
            response_tokens = 0
        
        total_time = time.time() - start_time
        
        # 创建调试结果
        debug_result = RAGDebugResult(
            question=question,
            timestamp=timestamp,
            retrieval_time=retrieval_time,
            llm_time=llm_time,
            total_time=total_time,
            retrieved_docs=retrieved_docs,
            context=context,
            final_answer=final_answer,
            retrieval_scores=retrieval_scores,
            embedding_vector=question_embedding,
            prompt_tokens=prompt_tokens,
            response_tokens=response_tokens
        )
        
        # 保存到历史记录
        self.debug_history.append(debug_result)
        
        if verbose:
            print("\n📊 性能总结:")
            print(f"  ⏱️ 总耗时: {total_time:.3f}秒")
            print(f"  🔍 检索耗时: {retrieval_time:.3f}秒 ({retrieval_time/total_time*100:.1f}%)")
            print(f"  🤖 LLM耗时: {llm_time:.3f}秒 ({llm_time/total_time*100:.1f}%)")
            print(f"  📊 Token使用: {prompt_tokens + response_tokens}")
            print("="*80)
        
        return debug_result
    
    def analyze_retrieval_quality(self, question: str, k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, Any]:
        """
        分析不同k值下的检索质量
        
        Args:
            question: 测试问题
            k_values: 不同的k值列表
            
        Returns:
            包含分析结果的字典
        """
        print(f"\n🧪 检索质量分析 - 问题: {question}")
        print("="*60)
        
        results = {}
        
        for k in k_values:
            print(f"\n📊 k={k} 的检索分析:")
            print("-"*30)
            
            try:
                docs_with_scores = self.vector_store.similarity_search_with_score(question, k=k)
                scores = [score for doc, score in docs_with_scores]
                
                # 计算统计信息
                avg_score = sum(scores) / len(scores) if scores else 0
                min_score = min(scores) if scores else 0
                max_score = max(scores) if scores else 0
                score_range = max_score - min_score if scores else 0
                
                results[k] = {
                    'num_docs': len(docs_with_scores),
                    'scores': scores,
                    'avg_score': avg_score,
                    'min_score': min_score,
                    'max_score': max_score,
                    'score_range': score_range,
                    'docs': [doc.page_content for doc, score in docs_with_scores]
                }
                
                print(f"  📄 文档数量: {len(docs_with_scores)}")
                print(f"  📊 平均分数: {avg_score:.4f}")
                print(f"  📈 分数范围: {min_score:.4f} - {max_score:.4f}")
                print(f"  📏 分数跨度: {score_range:.4f}")
                
                # 显示前3个文档的简要信息
                for i, (doc, score) in enumerate(docs_with_scores[:3]):
                    print(f"    {i+1}. 分数: {score:.4f}, 内容: {doc.page_content[:50]}...")
                    
            except Exception as e:
                print(f"  ❌ k={k} 检索失败: {e}")
                results[k] = {'error': str(e)}
        
        return results
    
    def compare_questions(self, questions: List[str], k: int = 4) -> pd.DataFrame:
        """
        比较多个问题的RAG性能
        
        Args:
            questions: 问题列表
            k: 检索文档数量
            
        Returns:
            包含比较结果的DataFrame
        """
        print(f"\n🔄 多问题性能比较 (k={k})")
        print("="*60)
        
        comparison_data = []
        
        for i, question in enumerate(questions):
            print(f"\n📝 问题 {i+1}: {question[:50]}...")
            
            result = self.comprehensive_debug(question, k=k, verbose=False)
            
            comparison_data.append({
                '问题': question[:50] + '...' if len(question) > 50 else question,
                '检索时间(秒)': result.retrieval_time,
                'LLM时间(秒)': result.llm_time,
                '总时间(秒)': result.total_time,
                '检索文档数': len(result.retrieved_docs),
                '平均相似度': sum(result.retrieval_scores) / len(result.retrieval_scores) if result.retrieval_scores else 0,
                '上下文长度': len(result.context),
                '答案长度': len(result.final_answer),
                'Prompt Tokens': result.prompt_tokens,
                'Response Tokens': result.response_tokens
            })
            
            print(f"  ⏱️ 总耗时: {result.total_time:.3f}秒")
            print(f"  📊 平均相似度: {sum(result.retrieval_scores) / len(result.retrieval_scores) if result.retrieval_scores else 0:.4f}")
        
        df = pd.DataFrame(comparison_data)
        print("\n📊 性能比较表:")
        print(df.to_string(index=False))
        
        return df
    
    def get_debug_summary(self) -> Dict[str, Any]:
        """
        获取调试历史的统计摘要
        
        Returns:
            包含统计信息的字典
        """
        if not self.debug_history:
            return {"message": "暂无调试历史"}
        
        total_queries = len(self.debug_history)
        avg_retrieval_time = sum(r.retrieval_time for r in self.debug_history) / total_queries
        avg_llm_time = sum(r.llm_time for r in self.debug_history) / total_queries
        avg_total_time = sum(r.total_time for r in self.debug_history) / total_queries
        avg_docs_retrieved = sum(len(r.retrieved_docs) for r in self.debug_history) / total_queries
        
        return {
            "总查询次数": total_queries,
            "平均检索时间": f"{avg_retrieval_time:.3f}秒",
            "平均LLM时间": f"{avg_llm_time:.3f}秒",
            "平均总时间": f"{avg_total_time:.3f}秒",
            "平均检索文档数": f"{avg_docs_retrieved:.1f}",
            "最近查询时间": self.debug_history[-1].timestamp
        }
    
    def export_debug_history(self, filename: str = None) -> str:
        """
        导出调试历史到JSON文件
        
        Args:
            filename: 文件名，如果为None则自动生成
            
        Returns:
            导出的文件路径
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rag_debug_history_{timestamp}.json"
        
        # 转换为可序列化的格式
        export_data = []
        for result in self.debug_history:
            export_data.append({
                "question": result.question,
                "timestamp": result.timestamp,
                "retrieval_time": result.retrieval_time,
                "llm_time": result.llm_time,
                "total_time": result.total_time,
                "num_retrieved_docs": len(result.retrieved_docs),
                "context_length": len(result.context),
                "final_answer": result.final_answer,
                "retrieval_scores": result.retrieval_scores,
                "prompt_tokens": result.prompt_tokens,
                "response_tokens": result.response_tokens
            })
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        print(f"📁 调试历史已导出到: {filename}")
        return filename

# 便捷函数
def quick_debug_rag(question: str, vector_store, embeddings, llm, prompt_template, k: int = 4):
    """
    快速调试RAG系统的便捷函数
    
    Args:
        question: 用户问题
        vector_store: 向量存储
        embeddings: 嵌入模型
        llm: 语言模型
        prompt_template: 提示模板
        k: 检索文档数量
    """
    debugger = RAGDebugger(vector_store, embeddings, llm, prompt_template)
    return debugger.comprehensive_debug(question, k=k)

def analyze_rag_performance(questions: List[str], vector_store, embeddings, llm, prompt_template, k: int = 4):
    """
    分析RAG系统在多个问题上的性能
    
    Args:
        questions: 问题列表
        vector_store: 向量存储
        embeddings: 嵌入模型
        llm: 语言模型
        prompt_template: 提示模板
        k: 检索文档数量
        
    Returns:
        性能分析DataFrame
    """
    debugger = RAGDebugger(vector_store, embeddings, llm, prompt_template)
    return debugger.compare_questions(questions, k=k)

if __name__ == "__main__":
    print("RAG调试工具集已加载")
    print("主要功能:")
    print("1. RAGDebugger - 完整的RAG调试器")
    print("2. quick_debug_rag - 快速调试函数")
    print("3. analyze_rag_performance - 性能分析函数")