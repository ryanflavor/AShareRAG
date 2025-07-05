from .ner_chinese import one_shot_ner_paragraph, one_shot_ner_output
from ...utils.llm_utils import convert_format_to_template

ner_conditioned_re_system = """您的任务是基于给定的中文段落和命名实体列表构建RDF（资源描述框架）图。
请以JSON格式返回三元组列表，每个三元组表示RDF图中的一个关系。

注意以下要求：
- 每个三元组应至少包含一个命名实体，最好包含两个实体
- 清楚地解析代词为具体名称以保持清晰度

"""


ner_conditioned_re_frame = """将段落转换为JSON字典，包含命名实体列表和三元组列表。
段落：
```
{passage}
```

{named_entity_json}
"""


ner_conditioned_re_input = ner_conditioned_re_frame.format(passage=one_shot_ner_paragraph, named_entity_json=one_shot_ner_output)


ner_conditioned_re_output = """{"triples": [
            ["综艺股份", "公司代码是", "600770"],
            ["综艺股份", "主营业务包括", "信息科技板块"],
            ["综艺股份", "主营业务包括", "新能源板块"],
            ["综艺股份", "主营业务包括", "股权投资板块"],
            ["南京天悦", "是子公司", "综艺股份"],
            ["南京天悦", "核心业务是", "助听器芯片"],
            ["南京天悦", "主要产品包括", "HA3950"],
            ["南京天悦", "主要产品包括", "HA330G"],
            ["南京天悦", "主要产品包括", "HA601SC"],
            ["南京天悦", "主要产品包括", "HA631SC"],
            ["神州龙芯", "是参股公司", "综艺股份"],
            ["神州龙芯", "核心业务是", "集成电路设计"],
            ["神州龙芯", "布局领域", "税控产品"],
            ["神州龙芯", "布局领域", "密码产品"],
            ["神州龙芯", "主要产品包括", "处理器"],
            ["神州龙芯", "主要产品包括", "主板"],
            ["毅能达", "是子公司", "综艺股份"],
            ["毅能达", "核心业务是", "智能卡"],
            ["毅能达", "主要产品包括", "磁条卡"],
            ["毅能达", "主要产品包括", "接触式IC卡"],
            ["毅能达", "主要产品包括", "感应式IC卡"],
            ["毅能达", "主要产品包括", "RFID卡"],
            ["毅能达", "主要产品包括", "智能穿戴手表"],
            ["毅能达", "主要产品包括", "自助终端设备"],
            ["掌上明珠", "是子公司", "综艺股份"],
            ["掌上明珠", "核心业务是", "手机移动游戏开发"],
            ["掌上明珠", "主要产品包括", "明珠三国"],
            ["掌上明珠", "主要产品包括", "明珠轩辕"],
            ["江苏综创", "是子公司", "综艺股份"],
            ["江苏综创", "核心业务是", "计算机系统集成"],
            ["综艺股份", "运营业务", "太阳能电站"],
            ["太阳能电站业务", "包括", "集中式光伏电站"],
            ["太阳能电站业务", "包括", "分布式光伏电站"],
            ["综艺光伏", "是子公司", "综艺股份"],
            ["综艺光伏", "核心业务是", "环保业务"],
            ["综艺光伏", "主要产品包括", "活性碳纤维材料"],
            ["综艺光伏", "主要产品包括", "VOCs吸附溶剂回收设备"],
            ["综艺光伏", "主要产品包括", "环保装备制造"],
            ["江苏高投", "是子公司", "综艺股份"],
            ["江苏高投", "核心业务是", "股权投资"]
    ]
}
"""


prompt_template = [
    {"role": "system", "content": ner_conditioned_re_system},
    {"role": "user", "content": ner_conditioned_re_input},
    {"role": "assistant", "content": ner_conditioned_re_output},
    {"role": "user", "content": convert_format_to_template(original_string=ner_conditioned_re_frame, placeholder_mapping=None, static_values=None)}
]