# 中文企业QA模板

one_shot_rag_qa_docs = (
    """公司名称: 聚飞光电\n聚飞光电股份有限公司成立于2005年，是一家专业从事LED封装及相关应用产品研发、生产和销售的高新技术企业。公司主要产品包括LED器件、背光源、照明产品等，在Micro LED和Mini LED技术方面有重要布局。\n"""
    """公司名称: 联得装备\n联得装备股份有限公司专注于显示面板生产设备的研发与制造，为液晶显示、OLED显示等提供专业设备解决方案。公司在Micro LED制造设备方面积累了丰富的技术经验。\n"""
    """公司名称: 明微电子\n明微电子股份有限公司是一家专业的LED驱动芯片供应商，为各类LED显示和照明应用提供高性能的驱动解决方案，包括为Micro LED显示提供专用驱动芯片。\n"""
)

one_shot_ircot_demo = (
    f"{one_shot_rag_qa_docs}"
    "\n\n问题: "
    f"哪些公司生产Micro LED相关产品？"
    "\n分析: "
    f"根据提供的信息，聚飞光电在Micro LED技术方面有重要布局，联得装备在Micro LED制造设备方面有技术积累，明微电子为Micro LED显示提供驱动芯片。Answer: 聚飞光电、联得装备、明微电子。"
    "\n\n"
)

rag_qa_system = (
    "作为一个专业的企业信息分析助手，你需要仔细分析提供的公司资料和相关问题。"
    '请在"分析："后详细说明你的推理过程，展示如何从文本中得出结论。'
    '最后用"Answer："给出简洁、准确的回答，专注于公司名称和相关业务信息。'
    "请用中文回答所有问题。"
)

one_shot_rag_qa_input = (
    f"{one_shot_rag_qa_docs}\n\n问题: 哪些公司生产Micro LED相关产品？\n分析: "
)

one_shot_rag_qa_output = (
    "根据提供的信息，聚飞光电在Micro LED技术方面有重要布局，联得装备在Micro LED制造设备方面有技术积累，明微电子为Micro LED显示提供驱动芯片。"
    "\nAnswer: 聚飞光电、联得装备、明微电子。"
)

prompt_template = [
    {"role": "system", "content": rag_qa_system},
    {"role": "user", "content": one_shot_rag_qa_input},
    {"role": "assistant", "content": one_shot_rag_qa_output},
    {"role": "user", "content": "${prompt_user}"},
]
