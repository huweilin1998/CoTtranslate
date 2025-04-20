import jieba
import re
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Optional, Any, Literal, Union
import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import as_completed
from tqdm import tqdm  # 用于显示进度条

class TranslationAssistant:
    def __init__(
            self,
            api_key: str,
            api_base: str,
            default_model: str = "deepseek-chat",
            default_mode: Literal["abstract_only", "with_main_text"] = "with_main_text"
    ):
        """
        Initialize Translation Assistant

        Args:
            api_key: OpenAI API key
            api_base: API base URL
            default_model: Default model to use for translation
            default_mode: Default processing mode ('abstract_only' or 'with_main_text')
        """
        self.api_key = api_key
        self.api_base = api_base
        self.default_model = default_model
        self.default_mode = default_mode
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        jieba.load_userdict('textile_keyword.txt')

    @staticmethod
    def clean_string(input_string: str) -> str:
        """Remove unstructured information from API response strings."""
        input_string = input_string.replace(', ', ',').replace('```', '').replace('json', '').replace('python', '').replace('result = ', '').replace('"词语": ', '')
        if not input_string.startswith('{'):
            input_string = input_string.replace(input_string.split('{')[0],'')
        if not input_string.endswith('}'):
            input_string = input_string.replace(input_string.split('}')[-1],'')
        return input_string

    @staticmethod
    def remove_brackets(text: str) -> str:
        """Remove all brackets and parentheses content from text."""
        pattern = re.compile(r'\[.*?\]|\(.*?\)')
        while re.search(pattern, text):
            text = re.sub(pattern, '', text)
        return text

    def translate_term(self, term: str, model: Optional[str] = None) -> str:
        """
        翻译单个术语
        Args:
            term: 要翻译的中文术语
            model: 指定使用的模型
        Returns:
            翻译后的英文术语
        """
        model = model or self.default_model
        prompt = f"请帮我将：{term}，翻译为英文，直接输出结果，不需要无关旁白。"

        try:
            response, _ = self.get_gpt_response(prompt, None, model)
            # 清理响应内容
            translated = response.strip().replace('"', '').replace("'", "")
            if ',' in translated:  # 如果返回多个结果，取第一个
                translated = translated.split(',')[0].strip()
            return translated
        except Exception as e:
            print(f"翻译术语 '{term}' 时出错: {str(e)}")
            return term  # 出错时返回原术语

    def batch_translate_terms(
            self,
            input_path: str = '纺织辞典.txt.txt',
            output_path: str = 'translated_terms.txt',
            model: Optional[str] = None,
            max_workers: int = 5
    ) -> Dict[str, str]:
        """
        批量翻译术语表
        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径
            model: 使用的模型
            max_workers: 最大线程数
        Returns:
            翻译结果字典 {中文术语: 英文翻译}
        """
        # 读取术语表
        with open(input_path, 'r', encoding='utf-8') as f:
            terms = [line.strip() for line in f if line.strip()]

        # 去除重复术语
        unique_terms = list(set(terms))
        print(f"开始翻译 {len(unique_terms)} 个唯一术语...")

        # 准备多线程任务
        tasks = [(term, model) for term in unique_terms]

        # 执行多线程翻译
        translated_dict = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.translate_term, term, model): term
                for term in unique_terms
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="翻译进度"):
                term = futures[future]
                try:
                    translated = future.result()
                    translated_dict[term] = translated
                except Exception as e:
                    print(f"术语 '{term}' 翻译失败: {str(e)}")
                    translated_dict[term] = term

        # 保存结果
        with open(output_path, 'w', encoding='utf-8') as f:
            for chinese, english in translated_dict.items():
                f.write(f"{chinese}\t{english}\n")

        print(f"翻译完成，结果已保存到 {output_path}")
        return translated_dict

    def get_gpt_response(
            self,
            prompt: str,
            info: Any = None,
            model: Optional[str] = None
    ) -> Tuple[str, Any]:
        """
        Get response from GPT model with additional info attached.

        Args:
            prompt: The prompt to send to the model
            info: Additional information to attach to the response
            model: Model to use (defaults to instance default_model)

        Returns:
            Tuple containing (response_content, original_info)
        """
        model = model or self.default_model
        completion = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=1
        )
        return completion.choices[0].message.content, info

    @staticmethod
    def restructure_dicts(input_dicts: Dict, template: Dict) -> Dict:
        """Restructure dictionaries according to the template."""
        # result = {key: {k: 0 for k in v} for key, v in template.items()}
        #
        # for input_dict in input_dicts.values():
        #     for template_key, template_values in template.items():
        #         if set(input_dict) & set(template_values):
        #             for key in template_values:
        #                 result[template_key][key] = input_dict.get(key, 0)
        # return result
        temp = {}
        temp2 = {}
        for part in list(template.keys()):
            temp['{}'.format(part.split('(')[0])] = part
            temp2['{}'.format(part.split('(')[0])] = template[part]
        result = {}
        for key,value in input_dicts.items():
            key = key.replace('[','').replace(']','')
            if key in temp.keys():
                new = {}
                for k,v in value.items():
                    if k in temp2[key]:
                        new[k] = v
                result[temp[key]] = new
        return result

    def generate_prompt(
            self,
            abstract: str,
            word_dict: Dict,
            mode: Optional[Literal["abstract_only", "with_main_text"]] = None,
            main_text: Optional[str] = None
    ) -> str:
        """
        Generate prompt for GPT based on content and word dictionary.

        Args:
            abstract: The abstract text
            word_dict: Dictionary of words and their possible translations
            mode: Processing mode ('abstract_only' or 'with_main_text')
            main_text: Main text content (required for 'with_main_text' mode)
        """
        mode = mode or self.default_mode

        if mode == "with_main_text":
            if not main_text:
                raise ValueError("Main text is required for 'with_main_text' mode")
            return self._generate_main_prompt(abstract, main_text, word_dict)
        else:
            return self._generate_basic_prompt(abstract, word_dict)

    def _generate_basic_prompt(self, abstract: str, word_dict: Dict) -> str:
        """Generate basic prompt for single abstract."""
        for key in word_dict:
            abstract = abstract.replace(key, f'[{key}]')

        prompt_parts = [
            '''{"任务":"请思考在对学术论文进行翻译时，[论文摘要]中[]应该被如何翻译为英文，对于每个[]中的词你有一个可选词语列表，你总共拥有100票请给可选词语列表中的术语投票，以Python字典形式输出结果，字典中又嵌套了多个字典，每个字典的键是'[词语]'，值是一个内嵌的字典，内嵌字典的键是术语，值是票数，即使票数为0也要输出，直接输出结果不要输出分析过程和旁白，在一行内输出。。",''',
            f'"论文摘要":"{abstract}",'
        ]

        for key, value in word_dict.items():
            key = key.split('(')[0]
            prompt_parts.append(f'"[{key}]":"{str(value).replace(", ", ",")}",')

        return ''.join(prompt_parts)[:-1] + '}'

    def _generate_main_prompt(self, abstract: str, main_text: str, word_dict: Dict) -> str:
        """Generate prompt including both abstract and main text."""
        main_text = self.remove_brackets(main_text)
        for key in word_dict:
            abstract = abstract.replace(key, f'[{key}]')
            main_text = main_text.replace(key, f'[{key}]')

        prompt_parts = [
            '''{"任务":"请思考在对学术论文进行翻译时，[论文摘要]中[]应该被如何翻译为英文，对于每个[]中的词你有一个可选词语列表，你总共拥有100票请给可选词语列表中的术语投票，以Python字典形式输出结果，字典中又嵌套了多个字典，每个字典的键是'[词语]'，值是一个内嵌的字典，内嵌字典的键是术语，值是票数，即使票数为0也要输出，直接输出结果不要输出分析过程和旁白，在一行内输出。",''',
            f'"论文摘要":"{abstract}",',
            f'"论文正文":"{main_text}",'
        ]

        for key, value in word_dict.items():
            key = key.split('(')[0]
            prompt_parts.append(f'"[{key}]":"{str(value).replace(", ", ",")}",')

        return ''.join(prompt_parts)[:-1] + '}'

    @staticmethod
    def create_word_database(use_first_field: bool) -> Dict:
        """Create word database from text file."""
        word_db = {}
        with open('word_db.txt', 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 3:
                    continue

                word = parts[0].lower()
                if word in word_db:
                    continue

                field_index = 1 if use_first_field else 2
                if parts[field_index]:
                    translations = eval(parts[field_index])
                    filtered = {k: v for k, v in translations.items() if v >= 0.01}
                    if len(filtered) > 1:
                        word_db[word] = filtered
        return word_db

    @staticmethod
    def get_max_value_key(dictionary: Dict) -> Optional[Any]:
        """Get key with maximum value from dictionary."""
        return max(dictionary, key=dictionary.get) if dictionary else None

    def process_abstract(
            self,
            abstract_data: str,
            mode: Optional[Literal["abstract_only", "with_main_text"]] = None,
            model: Optional[str] = None
    ) -> Tuple[Dict, float, Dict]:
        """
        Process a single abstract and return results with score and word_dict.

        Args:
            abstract_data: Input data containing tab-separated abstract, main text and translation
            mode: Processing mode ('abstract_only' or 'with_main_text')
            model: Model to use for this processing

        Returns:
            Tuple containing (restructured_dict, accuracy_score, word_dict)
        """
        mode = mode or self.default_mode
        model = model or self.default_model

        parts = abstract_data.split('\t')
        if len(parts) < 3:
            return {}, 0.0, {}

        abstract = parts[0]
        main_text = parts[1][:int(len(parts[1]) * 0.5)] if mode == "with_main_text" else None
        eabstract = parts[2]

        word_db = self.create_word_database(False)
        words = set(part for part in jieba.cut(abstract, cut_all=False) if part in word_db)

        word_dict = {}
        for word in words:
            count = sum(1 for other_word in words if word in other_word)
            if count == 1:
                matching_translations = [t for t in word_db[word] if t in eabstract]
                if matching_translations:
                    formatted_word = f"{word}({matching_translations[0]})"
                    word_dict[formatted_word] = list(word_db[word].keys())

        prompt = self.generate_prompt(abstract, word_dict, mode, main_text)
        response, _ = self.get_gpt_response(prompt, word_dict, model)

        try:
            cleaned_response = self.clean_string(response)
            result_data = eval(cleaned_response)
            restructured = self.restructure_dicts(result_data, word_dict)

            total = 0
            correct = 0
            for key, votes in restructured.items():
                if sum(votes.values()) > 0:
                    total += 1
                    if list(votes.keys())[0] == self.get_max_value_key(votes):
                        correct += 1

            score = correct / total if total > 0 else 0.0
            return restructured, round(score, 5), word_dict

        except Exception as e:
            print(f"Error processing response: {e}")
            return {}, 0.0, word_dict

    def execute_parallel_tasks(
            self,
            tasks: List[Tuple[str, Dict]],
            mode: Optional[Literal["abstract_only", "with_main_text"]] = None,
            model: Optional[str] = None,
            max_workers: int = 3
    ) -> List[Tuple[str, Union[Dict, str], Dict]]:
        """
        Execute tasks in parallel using ThreadPoolExecutor with info preservation.

        Args:
            tasks: List of tuples (prompt, info) to process
            mode: Processing mode ('abstract_only' or 'with_main_text')
            model: Model to use for all tasks
            max_workers: Maximum number of parallel workers

        Returns:
            List of tuples (function_name, result, original_info)
        """
        mode = mode or self.default_mode
        model = model or self.default_model

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                (executor.submit(self.get_gpt_response, task[0], task[1], model), task[1])
                for task in tasks
            ]

            results = []
            for future, info in futures:
                try:
                    response_content, original_info = future.result()
                    results.append(("get_gpt_response", response_content, original_info))
                except Exception as e:
                    print(f"Task failed: {e}")
                    results.append(("get_gpt_response", None, info))
            return results

    def run_pipeline(
            self,
            input_path: str,
            output_path: str,
            mode: Optional[Literal["abstract_only", "with_main_text"]] = None,
            model: Optional[str] = None,
            batch_size: int = 1
    ) -> List[Tuple[Dict, float, Dict]]:
        """
        Run the complete translation assistance pipeline with info preservation.

        Args:
            input_path: Path to input file
            output_path: Path to output file
            mode: Processing mode ('abstract_only' or 'with_main_text')
            model: Model to use for processing
            batch_size: Number of abstracts to process in each batch

        Returns:
            List of tuples (restructured_dict, accuracy_score, word_dict)
        """
        mode = mode or self.default_mode
        model = model or self.default_model

        with open(input_path, 'r', encoding='utf-8') as f:
            abstracts = [line.strip() for line in f if line.strip()]

        results = []

        for i in range(0, len(abstracts), batch_size):
            batch = abstracts[i:i + batch_size]

            # First prepare all prompts and word_dicts
            tasks = []
            word_dicts = []
            for abstract_data in batch:
                parts = abstract_data.split('\t')
                if len(parts) < 3:
                    continue

                abstract = parts[0]
                main_text = parts[1] if mode == "with_main_text" else None
                eabstract = parts[2]

                word_db = self.create_word_database(False)
                words = set(part for part in jieba.cut(abstract, cut_all=False) if part in word_db)

                word_dict = {}
                for word in words:
                    count = sum(1 for other_word in words if word in other_word)
                    if count == 1:
                        # 获取所有匹配的翻译术语并按长度降序排序
                        matching_translations = sorted(
                            [t for t in word_db[word] if t in eabstract],
                            key=lambda x: len(x),
                            reverse=True  # 按长度降序排列
                        )

                        if matching_translations:
                            # 取最长的匹配术语作为标准答案
                            best_translation = matching_translations[0]
                            formatted_word = f"{word}({best_translation})"
                            word_dict[formatted_word] = list(word_db[word].keys())

                prompt = self.generate_prompt(abstract, word_dict, mode, main_text)
                tasks.append((prompt, word_dict))
                word_dicts.append(word_dict)

            # Process in parallel
            batch_results = self.execute_parallel_tasks(tasks, mode, model)

            # Process results
            for (_, response_content, word_dict), original_word_dict in zip(batch_results, word_dicts):
                # try:
                try:
                    if not response_content:
                        with open(output_path, 'a', encoding='utf-8') as out:
                            out.write(f"\n")
                        continue
                    # print('*' * 100)
                    # print('初始输出',response_content)
                    cleaned_response = self.clean_string(response_content)
                    result_data = eval(cleaned_response)
                    # print('-'*100)
                    # print('处理输出',result_data)
                    # print('-' * 100)
                    # print('标准答案', original_word_dict)
                    # print('-' * 100)
                    restructured = self.restructure_dicts(result_data, original_word_dict)
                    # print('最终结果',restructured)

                    total = 0
                    correct = 0
                    for key, votes in restructured.items():
                        if sum(votes.values()) > 0:
                            total += 1
                            if list(votes.keys())[0] == self.get_max_value_key(votes):
                                correct += 1

                    score = correct / total if total > 0 else 0.0
                    results.append((restructured, round(score, 5), original_word_dict))

                    with open(output_path, 'a', encoding='utf-8') as out:
                        out.write(f"{restructured}\n")
                except:
                    with open(output_path, 'a', encoding='utf-8') as out:
                        out.write(f"{{}}\n")

                # except Exception as e:
                #     print(f"Error processing batch result: {e}")
                #     results.append(({}, 0.0, original_word_dict))

        return results

    def evaluate_results(
            self,
            results_path: str,
            q: float = 0.5,
            word_db_path: str = 'word_db.txt'
    ) -> float:
        """
        评估结果准确率，考虑原始翻译概率权重

        参数:
            results_path: 结果文件路径
            q: 权重(0-1)，原始翻译概率的权重
            word_db_path: 术语数据库路径

        返回:
            平均准确率(0-1)
        """
        # 创建术语数据库 {术语: {翻译: 概率}}
        word_db = {}
        with open(word_db_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3 and parts[2]:  # 确保有翻译概率字段
                    try:
                        translations = eval(parts[2])
                        if len(translations) > 1:  # 只保留有多选项的术语
                            word_db[parts[0]] = translations
                    except:
                        continue

        all_scores = []

        with open(results_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    data = eval(line)
                except:
                    continue  # 跳过格式错误的数据

                # 跳过空字典(大模型调用出错)
                if not data or data == {}:
                    continue

                score = 0
                valid_terms = 0

                for key, votes in data.items():
                    # 获取原始术语(去掉括号内容)
                    original_term = key.split('(')[0]

                    # 获取标准答案(括号内的内容)
                    standard_answers = key.split('(')[1].replace(')','')

                    # 检查是否有有效的标准答案和原始翻译概率
                    if not standard_answers or original_term not in word_db:
                        continue

                    valid_terms += 1

                    # 获取原始翻译概率
                    original_probs = word_db[original_term]

                    # 归一化投票结果(处理全0投票的情况)
                    total_votes = sum(votes.values())
                    if total_votes == 0:
                        # 全0投票时，默认选择第一个候选词
                        norm_votes = {k: 1.0 if i == 0 else 0.0
                                      for i, k in enumerate(votes.keys())}
                    else:
                        norm_votes = {k: v / total_votes for k, v in votes.items()}

                    # 归一化原始概率
                    total_original = sum(original_probs.values())
                    norm_original = {k: v / total_original for k, v in original_probs.items()}

                    # 加权合并结果
                    combined = {
                        k: q * norm_original.get(k, 0) + (1 - q) * norm_votes.get(k, 0)
                        for k in set(norm_original) | set(norm_votes)
                    }

                    # 获取最高票选项
                    best_translation = max(combined.items(), key=lambda x: x[1])[0]

                    # 检查是否匹配标准答案
                    if best_translation in standard_answers:
                        score += 1

                # 只有有效术语才计入统计
                if valid_terms > 0:
                    all_scores.append(score / valid_terms)

        # 计算平均准确率
        return round(sum(all_scores) / len(all_scores) *100, 1) if all_scores else 0.0

    def plot_precision_curves(
            self,
            result_files: List[str],
            labels: List[str],
            q_range: List[float] = None,
            ylim: Tuple[float, float] = (70, 93)
    ) -> None:
        """
        绘制不同q值下的准确率曲线

        参数:
            result_files: 结果文件路径列表
            labels: 对应的图例标签列表
            q_range: q值范围(默认0到1，步长0.1)
            ylim: y轴范围
        """

        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False

        # 设置默认q范围
        if q_range is None:
            q_range = np.arange(0, 1.1, 0.1)

        # 为每个结果文件计算不同q值下的准确率
        all_y = []
        for file in result_files:
            y = [
                self.evaluate_results(file, q=q)
                for q in q_range
            ]
            all_y.append(y)

        for index in range(len(labels)):
            print(labels[index],all_y[index])

        # 绘制曲线
        lines = []
        for y, label in zip(all_y, labels):
            line, = plt.plot(q_range, y, label=label, marker='o')
            lines.append(line)
            # self._annotate_extrema(q_range, y, line)

        # 设置图表属性
        plt.xlabel('q value')
        plt.ylabel('Precision（%）')
        plt.ylim(ylim)
        plt.legend()
        plt.grid(True)
        plt.title('Precision under Different q Values')
        plt.show()

    @staticmethod
    def _annotate_extrema(x: List[float], y: List[float], line) -> None:
        """在曲线上标注最大值和最小值"""
        max_idx = np.argmax(y)
        min_idx = np.argmin(y)

        plt.annotate(
            f'{y[max_idx]:.1f}',
            (x[max_idx], y[max_idx]),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center',
            color=line.get_color()
        )
        plt.annotate(
            f'{y[min_idx]:.1f}',
            (x[min_idx], y[min_idx]),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center',
            color=line.get_color()
        )

    def create_term_database(self) -> Tuple[Dict, Dict]:
        """
        创建术语数据库（已去除cxhy相关部分）
        Returns:
            tuple: (术语字典, 完整术语数据库)
        """
        term_dict = {}

        with open('word_db.txt', 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                term = parts[0].lower()
                if term not in term_dict:
                    term_dict[term] = list(eval(parts[-3]).keys())[0]
        return term_dict

    def com_dict(self, words: str, term_dict: Dict, format_output: bool = True) -> str:
        """
        构建术语词典字符串
        Args:
            words: 分号分隔的术语字符串
            term_dict: 术语字典
            format_output: 是否格式化输出
        """
        term_list = words.split(';')
        result = {}

        for term in term_list:
            if term in term_dict:
                result[term] = term_dict[term]

        return str(result) if format_output else result

    def gpt_plus_prompt(self, term_dict: Dict, words: str, abstract: str) -> str:
        """
        生成符合要求的提示词
        Args:
            term_dict: 术语字典
            words: 分号分隔的术语字符串
            abstract: 论文摘要
        """
        prompt_template = """{"任务":"请参考我给你的术语词典，帮我将论文中文摘要翻译成英文，不需要输出旁白等无关话语",
"术语词典": "%s",
"中文摘要": "%s"}"""

        return prompt_template % (
            self.com_dict(words, term_dict, True),
            abstract
        )

    def batch_retrieval_augmented_generation(
            self,
            input_path: str = 'input.txt',
            output_path: str = 'output.txt',
            model: str = None,
            max_workers: int = 10
    ) -> List[Dict]:
        """
        批量执行检索增强生成（保证输出顺序与输入一致）
        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径
            model: 使用的大模型
            max_workers: 最大线程数
        Returns:
            按原始顺序排列的结果列表
        """
        # 加载术语数据库
        term_dict = self.create_term_database()

        # 读取输入数据并保留原始顺序
        with open(input_path, 'r', encoding='utf-8') as f:
            inputs = [(i, line.strip().split('\t')[0]) for i, line in enumerate(f) if line.strip()]

        # 准备任务（保留原始行号）
        tasks = []
        for idx, abstract in inputs:
            # 提取术语
            terms = list(set(
                term for term in jieba.cut(abstract, cut_all=False)
                if term in term_dict
            ))
            words_str = ';'.join(terms)

            # 生成提示词
            prompt = self.gpt_plus_prompt(term_dict, words_str, abstract)
            print(prompt)
            print('*'*100)
            tasks.append((idx, prompt, abstract))

        # 多线程处理（但按顺序收集结果）
        results = [None] * len(tasks)  # 预分配结果列表

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_idx = {
                executor.submit(
                    self.get_gpt_response,
                    task[1],  # prompt
                    {'original': task[2], 'idx': task[0]},  # 保存原始文本和行号
                    model
                ): task[0]  # 用行号作为键
                for task in tasks
            }

            # 使用tqdm显示进度
            for future in tqdm(as_completed(future_to_idx), total=len(tasks), desc="处理进度"):
                idx = future_to_idx[future]
                try:
                    response, info = future.result()
                    results[idx] = {
                        'original': info['original'],
                        'enhanced': response.strip(),
                        'prompt': tasks[idx][1]
                    }
                except Exception as e:
                    print(f"第 {idx + 1} 行处理失败: {str(e)}")
                    results[idx] = {
                        'original': info['original'],
                        'enhanced': '',
                        'error': str(e)
                    }

        # 过滤可能的None值（理论上不应该出现）
        results = [r for r in results if r is not None]

        # 保存结果（按原始顺序）
        with open(output_path, 'w', encoding='utf-8') as f:
            for result in results:
                if 'error' in result:
                    f.write(f"Error: {result['error']}\n")
                else:
                    result['enhanced'] = result['enhanced'].replace('\n','')
                    f.write(f"{result['enhanced']}\n")

        return results

    def generate_basic_translation_prompt(self, abstract: str) -> str:
        """
        生成基础翻译提示词
        Args:
            abstract: 中文摘要文本
        Returns:
            格式化后的提示词
        """
        return '''{"任务":"帮我将论文中文摘要翻译成英文，不需要输出旁白等无关话语",
"中文摘要": "%s"}''' % abstract

    def batch_basic_translation(
            self,
            input_path: str = 'input.txt',
            output_path: str = 'output.txt',
            model: str = None,
            max_workers: int = 10
    ) -> List[Dict]:
        """
        批量执行基础翻译（保持顺序）
        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径
            model: 使用的大模型
            max_workers: 最大线程数
        Returns:
            按原始顺序排列的结果列表
        """
        # 读取输入数据并保留原始顺序
        with open(input_path, 'r', encoding='utf-8') as f:
            inputs = [(i, line.strip().split('\t')[0]) for i, line in enumerate(f) if line.strip()]

        # 准备任务（保留原始行号）
        tasks = []
        for idx, abstract in inputs:
            prompt = self.generate_basic_translation_prompt(abstract)
            tasks.append((idx, prompt, abstract))

        # 多线程处理（但按顺序收集结果）
        results = [None] * len(tasks)

        # print(tasks)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(
                    self.get_gpt_response,
                    task[1],  # prompt
                    {'original': task[2], 'idx': task[0]},  # 保存原始文本和行号
                    model
                ): task[0]
                for task in tasks
            }

            # 使用tqdm显示进度
            for future in tqdm(as_completed(future_to_idx), total=len(tasks), desc="翻译进度"):
                idx = future_to_idx[future]
                try:
                    response, info = future.result()
                    results[idx] = {
                        'original': info['original'],
                        'translated': response.strip(),
                        'prompt': tasks[idx][1]
                    }
                except Exception as e:
                    print(f"第 {idx + 1} 行翻译失败: {str(e)}")
                    results[idx] = {
                        'original': info['original'],
                        'translated': '',
                        'error': str(e)
                    }

        # 保存结果（按原始顺序）
        with open(output_path, 'w', encoding='utf-8') as f:
            for result in results:
                if result is not None:
                    if 'error' in result:
                        f.write(f"Error: {result['error']}\n")
                    else:
                        result['translated'] = result['translated'].replace('\n','')
                        f.write(f"{result['translated']}\n")

    def evaluate_translation_quality(
            self,
            hypothesis_path: str,
            source_path: str = None,
            detailed: bool = False
    ) -> Dict[str, float]:
        """
        评估翻译质量(BLEU分数)

        Args:
            hypothesis_path: 机器翻译结果文件路径
            reference_path: 参考翻译文件路径
            source_path: 源文本文件路径(可选)
            detailed: 是否返回详细评分

        Returns:
            包含评分结果的字典
        """
        try:
            import sacrebleu
        except ImportError:
            raise ImportError("请先安装sacrebleu: pip install sacrebleu")

        # 读取文件
        with open(hypothesis_path, 'r', encoding='utf-8') as hyp_f, \
                open(source_path, 'r', encoding='utf-8') as ref_f:

            hyp_lines = [line.strip() for line in hyp_f if line.strip()]
            ref_lines = [line.strip().split('\t')[-1] for line in ref_f if line.strip()]

        # 验证数据
        if len(hyp_lines) != len(ref_lines):
            raise ValueError(f"行数不匹配: 机器翻译{len(hyp_lines)}行, 参考翻译{len(ref_lines)}行")

        # 计算BLEU分数
        bleu = sacrebleu.corpus_bleu(hyp_lines, [ref_lines])

        # 构建结果
        result = {
            'bleu_score': round(bleu.score, 1),
        }

        # 打印结果
        print("\n" + "=" * 50)
        print(f" BLEU 评估结果 (N={len(hyp_lines)})")
        print(f" BLEU分数: {result['bleu_score']}")
        print("=" * 50 + "\n")

        return result

    def evaluate_term_translation_accuracy(
            self,
            input_path: str = 'input.txt',
            output_path: str = 'output.txt',
            term_db_path: str = 'word_db.txt',
    ) -> Dict[str, float]:
        """
        评估术语翻译准确率
        Args:
            input_path: 输入文件路径(包含中英文对照)
            output_path: 翻译结果文件路径
            term_db_path: 术语数据库路径
            min_term_length: 最小术语长度(避免短词误判)
        Returns:
            包含评估结果的字典
        """
        # 加载术语数据库
        term_dict = self.create_term_database()

        # 读取输入数据和翻译结果
        with open(input_path, 'r', encoding='utf-8') as f_in, \
                open(output_path, 'r', encoding='utf-8') as f_out:

            inputs = [line.strip().split('\t')[0] for line in f_in if line.strip()]
            translations = [line.strip() for line in f_out if line.strip()]

        if len(inputs) != len(translations):
            raise ValueError(f"行数不匹配: 输入{len(inputs)}行, 输出{len(translations)}行")

        result = []
        for index in range(len(inputs)):
            # 提取术语
            terms = {}
            for term in jieba.cut(inputs[index], cut_all=False):
                if term in inputs[index] and term in term_dict.keys():
                    terms[term] = term_dict[term].lower()

            pre = 0
            for key,value in terms.items():
                if value.lower() in translations[index].lower():
                    pre += 1
            result.append([round(pre/len(terms)*100,1),len(translations[index].split(' ')),len(terms)])
        print('准确率',round(sum([p[0] for p in result])/len(result),1))
        print('平均词数', round(sum([p[1] for p in result]) / len(result), 1))
        print('平均增强数', round(sum([p[2] for p in result]) / len(result), 1))
        print('*'*100)
        # 加载术语数据库

# Usage example with info preservation
if __name__ == "__main__":
    API_KEY = "sk-KIsbVOoWFKSzHJOHD11f309d6e9a4b36913c774290287b8d"
    API_BASE = "https://api.b3n.fun/v1"

    # Initialize with default settings
    assistant = TranslationAssistant(
        api_key=API_KEY,
        api_base=API_BASE,
    )
    for model in ['gpt-4o-mini', 'claude-3-7-sonnet-all', 'grok-3', 'gemini-2.0-flash', 'deepseek-chat'][3:]:
        assistant.run_pipeline(
            input_path='input.txt',
            mode="abstract_only",
            model=model,
            output_path='abstract_{}_10.txt'.format(model),
            batch_size=20
        )
        print(model,assistant.evaluate_results(
            results_path='abstract_{}_10.txt'.format(model),
            q=0,
        ))

    # 定义要比较的结果文件和对应标签
    # start = 'main_'
    # result_files = [
    #     start + 'gpt_4o_mini.txt',
    #     start + 'gemini_2.0_flash.txt',
    #     start + 'deepseek_chat.txt',
    #     start + 'claude_3_7_sonnet_all.txt',
    #     start + 'grok_3.txt'
    # ]
    # labels = [
    #     'gpt_4o_mini'.replace('_','-'),
    #     'gemini_2.0_flash'.replace('_','-'),
    #     'deepseek_chat'.replace('_','-'),
    #     'claude_3_7_sonnet_all'.replace('_', '-'),
    #     'grok_3'.replace('_', '-'),
    # ]
    #
    # # 绘制曲线
    # assistant.plot_precision_curves(
    #     result_files=result_files,
    #     labels=labels,
    #     ylim=(66, 78)  # 可调整y轴范围
    # )

    # assistant.batch_translate_terms(
    #     input_path="纺织辞典.txt",
    #     output_path="gpt-4o-mini.txt",
    #     model="gpt-4o-mini",
    #     max_workers=20
    # )

    # assistant.batch_retrieval_augmented_generation(
    #     input_path="input.txt",
    #     output_path="gpt-4o-mini.txt",
    #     model="gpt-4o-mini",
    #     max_workers=1
    # )

    # for model in ['gpt-4o-mini','claude-3-7-sonnet-all','grok-3','gemini-2.0-flash','deepseek-chat']:
    #     # assistant.batch_retrieval_augmented_generation(
    #     #     input_path="input.txt",
    #     #     output_path="RAG_{}.txt".format(model),
    #     #     model="{}".format(model),
    #     #     max_workers=20
    #     # )
    #     print(model)
    #     assistant.evaluate_translation_quality(
    #         hypothesis_path="RAG_{}.txt".format(model),
    #         source_path="input.txt",
    #         detailed=True
    #     )
    #     assistant.evaluate_term_translation_accuracy(
    #         input_path="input.txt",
    #         output_path="RAG_{}.txt".format(model),
    #         term_db_path="word_db.txt",
    #     )
