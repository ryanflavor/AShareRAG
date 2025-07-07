# **Workflow Plan: Brownfield Full-Stack Enhancement**

* **创建日期**: 2025年7月5日
* **项目**: 上市公司关联性RAG项目
* **类型**: 棕地 (Brownfield)
* **状态**: Active

## **目标 (Objective)**

对一个基于`hipporag`的现有原型进行重构和增强，解决知识图谱构建不完整的问题，并扩展其功能，使其支持“事实问答”和“关联性发现”两种查询模式。

## **选定的工作流 (Selected Workflow)**

* **工作流**: `brownfield-fullstack`
* **理由**: 项目属于在现有框架基础上的复杂功能增强，涉及完整的后端数据处理和API服务，此工作流覆盖了从分析、规划、设计到开发的完整生命周期。

## **工作流步骤 (Workflow Steps)**

### **第一阶段：规划与设计 (已完成)**
* [x] **步骤 1: 项目分析与需求澄清**
    * **代理**: Analyst (Mary), Product Manager (John)
    * **行动**: 分析现有原型和资产，与用户共同探索和明确了双功能需求、非功能性需求和风险。
    * **产出**: 一份全面、清晰的需求文档（PRD V1.2）。
* [x] **步骤 2: 架构设计**
    * **代理**: Architect (Winston)
    * **行动**: 基于PRD，设计了包含“双存储核心”、“适配器模式”、“TDD工作流”和“CI/CD策略”在内的完整技术架构。
    * **产出**: 一份详尽的、可执行的架构文档（Architecture Document V1.2）。
* [x] **步骤 3: 最终审核**
    * **代理**: Product Owner (Sarah)
    * **行动**: 使用`po-master-checklist`对PRD和架构文档进行了全面审核，确保两者的一致性与完整性。
    * **产出**: 批准项目进入开发阶段的最终决策。

### **第二阶段：开发与执行 (待办)**
* [x] **步骤 4: 文档分片 (Document Sharding)**
    * **代理**: Product Owner (PO)
    * **行动**: 在IDE环境中，将最终的PRD和架构文档分片成更小的、易于处理的Markdown文件，以便AI智能体在创建故事时能精确引用。
    * **产出**: `docs/prd/` 和 `docs/architecture/` 目录及其中文件。
* [ ] **步骤 5: 故事开发循环 (Story Development Cycle)** - *此步骤将重复执行*
    * [x] **a. 创建故事**: 由PO/SM代理根据分片文档创建具体的故事文件（Story File）。<!-- current-step: Story 2.3 created -->
    * [ ] **b. 开发实现**: 由Dev Agent根据TDD流程实现故事功能。
    * [ ] **c. 代码审核**: (可选) 由QA Agent进行代码质量和功能审核。
* [ ] **步骤 6: 史诗回顾 (Epic Retrospective)** - *可选*
    * **代理**: Product Owner (PO)
    * **行动**: 在每个Epic完成后，进行回顾，总结经验。

## **关键决策点 (Key Decisions Made)**

1.  **架构模式**: 确定采用“封装与注入”模式，将`hipporag`作为库使用。
2.  **存储选型**: 确定采用`igraph`+`LanceDB`的“双存储核心”。
3.  **功能范围**: 确定支持“事实问答”和“关联性发现”两种模式，并通过“意图路由器”分发。
4.  **开发方法**: 确定采用强制性的“测试驱动开发（TDD）”工作流。
5.  **部署策略**: 确定采用基于GitHub Actions的CI/CD流程。

## **预期产出物 (Expected Outputs)**

### **规划文档 (已完成)**
* [x] `prd.md` (V1.2)
* [x] `architecture.md` (V1.2)

### **开发产物 (待办)**
* [ ] `docs/stories/` 目录下的多个故事文件。
* [ ] `src/` 目录下的完整项目源代码。
* [ ] `tests/` 目录下的完整测试用例。
* [ ] 最终部署的API服务。

---