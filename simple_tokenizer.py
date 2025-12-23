#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
手写字节级BPE Tokenizer

这个实现展示了现代LLM tokenizer的核心机制：
1. Unicode文本 -> UTF-8 bytes（字节序列）
2. Bytes -> 通过BPE合并成tokens

为什么理解这个很重要？
- 很多模型的"莫名其妙的bug"其实是tokenizer问题
- 特殊字符、多语言文本、emoji的处理都依赖于正确的tokenizer
- 理解tokenization有助于prompt engineering和debug

作者注：本实现参考了GPT-2/GPT-4的字节级BPE设计
"""

from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import re


class SimpleBPETokenizer:
    """
    一个从零实现的字节级BPE Tokenizer
    
    核心概念：
    - 字节级(Byte-level)：直接在UTF-8字节上操作，而非字符
    - BPE：迭代地合并最频繁的字节对，形成更长的token
    
    为什么用字节级？
    1. 任何Unicode字符都能被编码（不会有OOV问题）
    2. 词表大小可控（基础词表只有256个字节）
    3. GPT系列模型都用这种方式
    """
    
    def __init__(self, vocab_size: int = 500):
        """
        初始化tokenizer
        
        Args:
            vocab_size: 目标词表大小，包含256个基础字节token
                        实际会进行 vocab_size - 256 次合并操作
        """
        self.vocab_size = vocab_size
        
        # 词表：token_id -> bytes
        # 初始化为256个基础字节 (0x00 - 0xFF)
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        
        # 合并规则：(byte1, byte2) -> merged_token_id
        # 这个顺序很重要！编码时必须按训练时的顺序应用合并
        self.merges: Dict[Tuple[int, int], int] = {}
        
        # 特殊token（可扩展）
        self.special_tokens: Dict[str, int] = {}
        
        # 用于分词的正则表达式（类似GPT-2的预分词）
        # 这个正则把文本切成更小的块，避免跨词合并
        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w\d]+|\s+(?!\S)|\s+""",
            re.UNICODE
        )
    
    # ==================== Unicode与Bytes的关系 ====================
    
    def _explain_unicode_to_bytes(self, text: str) -> None:
        """
        【教学方法】解释Unicode文本如何变成UTF-8字节
        
        UTF-8编码规则：
        - 1字节: 0xxxxxxx (ASCII, 0-127)
        - 2字节: 110xxxxx 10xxxxxx (拉丁扩展、希腊等)
        - 3字节: 1110xxxx 10xxxxxx 10xxxxxx (中日韩、大多数语言)
        - 4字节: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx (emoji、古文字)
        
        这就是为什么：
        - 英文token通常短，中文token通常长
        - 同样的vocab_size，英文能表示更多"概念"
        """
        print(f"\n{'='*60}")
        print(f"Unicode -> UTF-8 Bytes 解析: '{text}'")
        print(f"{'='*60}")
        
        for char in text:
            code_point = ord(char)  # Unicode码点
            utf8_bytes = char.encode('utf-8')  # UTF-8字节
            
            # 判断字节数
            if code_point < 0x80:
                byte_type = "1字节 (ASCII)"
            elif code_point < 0x800:
                byte_type = "2字节"
            elif code_point < 0x10000:
                byte_type = "3字节"
            else:
                byte_type = "4字节 (emoji/稀有)"
            
            print(f"  '{char}' | U+{code_point:04X} | {byte_type}")
            print(f"       UTF-8: {list(utf8_bytes)} -> {[hex(b) for b in utf8_bytes]}")
    
    # ==================== BPE 训练 ====================
    
    def _get_stats(self, token_ids_list: List[List[int]]) -> Dict[Tuple[int, int], int]:
        """
        统计所有相邻token对的出现频率
        
        这是BPE的核心：找到最频繁的相邻对
        
        Args:
            token_ids_list: 多个token序列的列表
        
        Returns:
            {(token1, token2): count} 的字典
        """
        stats = defaultdict(int)
        for token_ids in token_ids_list:
            for i in range(len(token_ids) - 1):
                pair = (token_ids[i], token_ids[i + 1])
                stats[pair] += 1
        return stats
    
    def _merge(self, token_ids: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
        """
        在token序列中执行一次合并操作
        
        例如: [1, 2, 3, 2, 3] + merge(2,3)->99 = [1, 99, 99]
        
        Args:
            token_ids: 原始token序列
            pair: 要合并的token对 (a, b)
            new_id: 合并后的新token id
        
        Returns:
            合并后的新序列
        """
        new_tokens = []
        i = 0
        while i < len(token_ids):
            # 检查当前位置是否匹配要合并的pair
            if (i < len(token_ids) - 1 and 
                token_ids[i] == pair[0] and 
                token_ids[i + 1] == pair[1]):
                new_tokens.append(new_id)
                i += 2  # 跳过两个token
            else:
                new_tokens.append(token_ids[i])
                i += 1
        return new_tokens
    
    def train(self, texts: List[str], verbose: bool = True) -> None:
        """
        在给定文本上训练BPE模型
        
        BPE训练算法：
        1. 将所有文本转换为UTF-8字节序列
        2. 统计所有相邻字节对的频率
        3. 合并最频繁的字节对，生成新token
        4. 重复步骤2-3，直到达到目标vocab_size
        
        Args:
            texts: 训练文本列表
            verbose: 是否打印训练过程
        """
        if verbose:
            print("\n" + "="*60)
            print("开始BPE训练")
            print("="*60)
        
        # Step 1: 预分词 + 转换为字节序列
        # 预分词的作用：避免跨词边界的合并（如 "dog" + " cat" 不应合并）
        token_ids_list: List[List[int]] = []
        
        for text in texts:
            # 使用正则预分词
            chunks = self.pat.findall(text)
            for chunk in chunks:
                # 每个chunk转换为UTF-8字节序列
                # 此时每个字节就是一个token (0-255)
                utf8_bytes = chunk.encode('utf-8')
                token_ids_list.append(list(utf8_bytes))
        
        if verbose:
            total_tokens = sum(len(ids) for ids in token_ids_list)
            print(f"预分词后共有 {len(token_ids_list)} 个chunk")
            print(f"初始token数量: {total_tokens}")
        
        # Step 2-4: 迭代合并
        num_merges = self.vocab_size - 256  # 需要进行的合并次数
        
        for i in range(num_merges):
            # 统计当前所有相邻对的频率
            stats = self._get_stats(token_ids_list)
            
            if not stats:
                if verbose:
                    print(f"没有更多可合并的pair，停止于 {i} 次合并")
                break
            
            # 找到最频繁的pair
            best_pair = max(stats, key=stats.get)
            best_count = stats[best_pair]
            
            # 分配新的token id
            new_id = 256 + i
            
            # 更新词表和合并规则
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            self.merges[best_pair] = new_id
            
            # 在所有序列中执行这次合并
            token_ids_list = [self._merge(ids, best_pair, new_id) for ids in token_ids_list]
            
            if verbose and (i < 10 or i % 50 == 0):
                # 尝试解码以显示合并的是什么
                try:
                    decoded = self.vocab[new_id].decode('utf-8', errors='replace')
                except:
                    decoded = repr(self.vocab[new_id])
                print(f"合并 #{i+1}: {best_pair} -> {new_id} "
                      f"(出现{best_count}次) = '{decoded}'")
        
        if verbose:
            final_tokens = sum(len(ids) for ids in token_ids_list)
            print(f"\n训练完成！词表大小: {len(self.vocab)}")
            print(f"Token数量: {total_tokens} -> {final_tokens} "
                  f"(压缩率: {total_tokens/final_tokens:.2f}x)")
    
    # ==================== 编码（文本 -> Token IDs） ====================
    
    def encode(self, text: str, verbose: bool = False) -> List[int]:
        """
        将文本编码为token id序列
        
        编码过程：
        1. 预分词（可选，用正则切分）
        2. 转换为UTF-8字节
        3. 按训练时的顺序应用所有合并规则
        
        Args:
            text: 输入文本
            verbose: 是否打印编码过程
        
        Returns:
            token id列表
        """
        if not text:
            return []
        
        # 预分词
        chunks = self.pat.findall(text)
        all_token_ids = []
        
        for chunk in chunks:
            # 转换为字节（初始token）
            token_ids = list(chunk.encode('utf-8'))
            
            if verbose:
                print(f"\nChunk: '{chunk}'")
                print(f"  UTF-8 bytes: {token_ids}")
            
            # 按合并顺序应用规则
            # 关键点：必须按训练时的顺序！
            # 这就是为什么merges要记录顺序
            for pair, new_id in self.merges.items():
                token_ids = self._merge(token_ids, pair, new_id)
            
            if verbose:
                print(f"  合并后: {token_ids}")
            
            all_token_ids.extend(token_ids)
        
        return all_token_ids
    
    # ==================== 解码（Token IDs -> 文本） ====================
    
    def decode(self, token_ids: List[int]) -> str:
        """
        将token id序列解码为文本
        
        解码过程：
        1. 每个token id查表得到bytes
        2. 拼接所有bytes
        3. 用UTF-8解码为字符串
        
        这里有个坑：
        - 如果token被切断在UTF-8序列中间，解码会出错
        - 这就是为什么有些模型在处理多语言时会出bug
        
        Args:
            token_ids: token id列表
        
        Returns:
            解码后的文本
        """
        # 拼接所有bytes
        all_bytes = b''.join(self.vocab.get(id, b'') for id in token_ids)
        
        # UTF-8解码（errors='replace'处理无效序列）
        return all_bytes.decode('utf-8', errors='replace')
    
    def decode_tokens(self, token_ids: List[int]) -> List[str]:
        """
        将每个token单独解码，用于调试
        
        这个方法能帮你看清每个token代表什么
        """
        result = []
        for id in token_ids:
            token_bytes = self.vocab.get(id, b'')
            try:
                decoded = token_bytes.decode('utf-8')
            except UnicodeDecodeError:
                # UTF-8不完整，显示原始字节
                decoded = repr(token_bytes)
            result.append(decoded)
        return result
    
    # ==================== 调试工具 ====================
    
    def analyze_tokenization(self, text: str) -> None:
        """
        深入分析一段文本的tokenization过程
        
        这个方法能帮你理解：
        - 为什么某些文本token数特别多
        - 为什么模型在某些字符上表现奇怪
        """
        print("\n" + "="*60)
        print(f"Tokenization分析: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        print("="*60)
        
        # 先展示Unicode到bytes的转换
        self._explain_unicode_to_bytes(text[:20] if len(text) > 20 else text)
        
        # 编码
        token_ids = self.encode(text)
        decoded_tokens = self.decode_tokens(token_ids)
        
        print(f"\n总Token数: {len(token_ids)}")
        print(f"字符数: {len(text)}")
        print(f"Token/字符比: {len(token_ids)/len(text):.2f}")
        
        print(f"\nToken详情:")
        for i, (id, token) in enumerate(zip(token_ids, decoded_tokens)):
            # 显示原始bytes
            token_bytes = self.vocab.get(id, b'')
            print(f"  [{i}] id={id:4d} | '{token}' | bytes={list(token_bytes)}")
        
        # 验证解码
        decoded = self.decode(token_ids)
        if decoded == text:
            print("\n✓ 编解码验证通过")
        else:
            print(f"\n✗ 编解码不匹配!")
            print(f"  原文: {repr(text)}")
            print(f"  解码: {repr(decoded)}")


# ==================== 特殊情况演示 ====================

def demonstrate_tokenization_issues():
    """
    演示常见的tokenization问题
    
    这些问题就是很多"模型bug"的真正原因
    """
    print("\n" + "="*70)
    print("常见Tokenization问题演示")
    print("="*70)
    
    # 问题1: 中英文token长度差异
    print("\n【问题1】中英文Token长度差异")
    print("-"*40)
    en_text = "hello world"
    zh_text = "你好世界"
    
    print(f"英文 '{en_text}':")
    print(f"  字符数: {len(en_text)}")
    print(f"  UTF-8字节数: {len(en_text.encode('utf-8'))}")
    
    print(f"中文 '{zh_text}':")
    print(f"  字符数: {len(zh_text)}")
    print(f"  UTF-8字节数: {len(zh_text.encode('utf-8'))}")
    print("  结论: 同样4个字符，中文需要3倍的字节，意味着更多的token")
    
    # 问题2: Emoji和特殊字符
    print("\n【问题2】Emoji和特殊字符")
    print("-"*40)
    emoji_text = "😀🎉"
    print(f"Emoji '{emoji_text}':")
    for emoji in emoji_text:
        utf8 = emoji.encode('utf-8')
        print(f"  '{emoji}' = {len(utf8)}字节 = {list(utf8)}")
    print("  结论: 每个emoji需要4字节，可能需要多个token表示")
    
    # 问题3: 空格和换行的诡异行为
    print("\n【问题3】空格和换行的token化")
    print("-"*40)
    texts = ["hello", " hello", "  hello", "hello\n", "hello\t"]
    for t in texts:
        print(f"  {repr(t):15s} -> bytes: {list(t.encode('utf-8'))}")
    print("  结论: 前导空格、多个空格、换行符都是独立token，影响模型理解")
    
    # 问题4: 数字的切分
    print("\n【问题4】数字的token化")
    print("-"*40)
    numbers = ["123", "1234", "12345", "123456789"]
    print("  数字可能被切成奇怪的组合:")
    for n in numbers:
        utf8 = list(n.encode('utf-8'))
        print(f"  '{n}' -> bytes: {utf8}")
    print("  结论: 大数字可能被切成多个token，影响数学推理")


def main():
    """主函数：演示完整的tokenizer工作流程"""
    
    # 训练数据（实际应用中应该用更大的语料）
    training_texts = [
        "Hello world! This is a simple BPE tokenizer.",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning and natural language processing are fascinating.",
        "你好世界！这是一个简单的BPE分词器。",
        "深度学习改变了人工智能的发展方向。",
        "Python is a great programming language for AI development.",
        "Tokenization is crucial for understanding language models.",
        "Special characters like @#$% need careful handling.",
        "Numbers like 12345 and dates like 2024-01-15 are tricky.",
        "Emojis 😀🎉 are encoded as multiple bytes in UTF-8.",
    ]
    
    # 创建并训练tokenizer
    tokenizer = SimpleBPETokenizer(vocab_size=350)  # 256 + 94次合并
    tokenizer.train(training_texts, verbose=True)
    
    # 测试编解码
    print("\n" + "="*60)
    print("测试编解码")
    print("="*60)
    
    test_texts = [
        "Hello world!",
        "你好世界！",
        "Machine learning is cool 😀",
        "Test 12345 numbers",
    ]
    
    for text in test_texts:
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        print(f"\n原文: '{text}'")
        print(f"Token IDs: {tokens}")
        print(f"Token数: {len(tokens)}")
        print(f"解码: '{decoded}'")
        print(f"匹配: {'✓' if text == decoded else '✗'}")
    
    # 详细分析
    tokenizer.analyze_tokenization("Hello 你好 😀")
    
    # 演示常见问题
    demonstrate_tokenization_issues()
    
    print("\n" + "="*60)
    print("总结：为什么tokenizer问题会导致模型bug？")
    print("="*60)
    print("""
1. 【上下文窗口问题】
   - 中文每字符占3字节，英文只占1字节
   - 同样的token限制，中文能容纳的内容更少
   
2. 【数学推理困难】
   - 数字被拆分成奇怪的token组合
   - 模型难以理解123和12、3的关系

3. 【特殊字符乱码】
   - emoji、罕见字符可能被错误切分
   - 导致生成时出现乱码或截断

4. 【空格敏感性】
   - "hello"和" hello"是完全不同的token序列
   - 导致prompt微小变化产生不同结果

5. 【多语言不公平】
   - 某些语言需要更多token表示相同含义
   - 模型在这些语言上性能会下降

理解这些，就能更好地debug模型行为！
""")


if __name__ == "__main__":
    main()
