# 3. 非功能性需求 (Non-Functional Requirements)

* **NFR1 (性能 - Performance):**
    * 系统的设计应优先保证**答案的准确性和相关性**，其次才是响应速度。系统应尽可能返回所有高度相关的公司，而不应为了速度牺牲结果的全面性。

* **NFR2 (准确性与相关性 - Accuracy & Relevance):**
    * 系统定义“相关性”时，必须遵循以下优先级层次：
        1.  **最高优先级**：在**细分业务**上直接生产竞争产品的公司。
        2.  **中优先级**：生产相似或可替代产品的公司。
        3.  **低优先级**：处于产业链上下游关系的公司。
    * 系统的相似度分析必须具备**业务分类的深度感知能力**，能够深入到具体的、细分的业务层面进行比较。

* **NFR3 (可维护性 - Maintainability):**
    * **a. 增量更新与新增**：当数据发生变化或有新公司加入时，系统必须支持**增量索引**。
    * **b. 精确删除**：系统必须提供一个机制，能够根据唯一标识符**精确地删除**该公司所有相关的索引数据。
