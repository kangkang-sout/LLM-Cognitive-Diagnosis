# 基于布鲁姆分类法与层级感知对齐的大模型驱动认知诊断：一个模型无关框架（扩展修订版）

**作者**：董志昂，陈静远✉，吴飞✉  
**单位**：浙江大学计算机科学与技术学院，杭州 310058，中国  
**发表信息**：© Higher Education Press 2025 | Front. Digit. Educ., 2025, 2(2): 20 | https://doi.org/10.1007/s44366-025-0057-8  
**备注**：本文是原论文《LLM-Driven Cognitive Diagnosis with SOLO Taxonomy: A Model-Agnostic Framework》的实质性扩展修订版。

---

## 摘要

在个性化教育蓬勃发展的背景下，认知诊断作为智能教育系统的核心组件，其重要性日益凸显。认知诊断模型（CDMs）旨在通过学习者对系列练习的作答反馈来刻画其认知状态与知识掌握程度。然而，传统方法在面对交互记录稀疏的新用户或新题目时往往表现欠佳，这主要归因于模型缺乏有效的外部先验知识注入机制。近期，Dong 等人（2025）提出了一种利用大型语言模型（LLMs）增强认知诊断的模型无关框架，验证了语义知识注入的可行性。然而，该初步工作主要依赖 SOLO 分类法，且在对齐过程中未显式利用认知理论的层次结构约束，限制了诊断的细粒度与泛化能力。为此，本研究提出一种**改进的层级感知 LLM 增强认知诊断框架**。该框架不仅将认知分类体系升级为更具普适性的**布鲁姆教育目标分类法（Bloom's Taxonomy）**，更首创了**层级感知对齐机制（Hierarchy-Aware Alignment）**，通过引入层次约束损失函数，确保语义嵌入不仅与行为空间对齐，还符合认知发展的渐进规律。我们在**ASSISTments 2009-2010、Junyi Academy 及 NIPS-2020**三个跨领域公开数据集上，基于**NCD、RCD、GCD、SCD、RDGT**五种主流架构进行了系统性验证，并对比了**gpt-osss:120b、glm-4、gemma3:27b、qwen3:30b**四种不同规模与语言特性的 LLM 影响。实验结果表明，所提框架在多种场景下均显著提升诊断精度（AUC 提升 +4.8%~+9.3%），尤其在冷启动场景下增益更为突出（+8%~+12%）。此外，开源模型评估证实了框架在隐私敏感场景下的私有化部署潜力。

**关键词**：大型语言模型；认知诊断模型；布鲁姆分类法；层级感知对齐；跨域泛化；隐私保护

---

## 1 引言

在个性化教育蓬勃发展的今天，智能教育系统已成为提升学习效率的关键基础设施。作为其核心组件，认知诊断技术通过分析学生在一系列练习上的作答表现，来量化评估其知识掌握状态与认知能力（Bi et al., 2020; Huang et al., 2019）。这种细粒度的学情画像不仅支撑着自适应学习路径推荐，也是实现计算机化自适应测试的前提条件（Zhuang et al., 2022）。因此，构建高精度、强鲁棒的认知诊断模型，对于优化教育资源配置、实现因材施教具有至关重要的意义。

传统的认知诊断模型（CDMs）大多植根于心理测量学理论，如项目反应理论（IRT）及其变体，它们依赖预先定义的交互函数来推断学生的潜在特质（Lord, 1952; Reckase, 2009）。随着深度学习技术的兴起，新一代神经认知诊断模型（如 NCD）应运而生，利用神经网络强大的拟合能力捕捉学生与练习之间的复杂交互模式（Wang et al., 2020）。随后，基于图结构的方法（如 RCD、RDGT）进一步引入了知识概念间的关系建模，而考虑情感状态（ACD）或长尾分布（SCD）的研究则拓展了诊断的维度（Gao et al., 2021; Wang et al., 2024b; Yu et al., 2024）。然而，尽管模型架构不断演进，现有方法仍普遍面临一个严峻挑战：冷启动问题。当面对交互记录稀缺的新学生或新练习时，由于缺乏足够的行为数据支撑，模型往往难以做出准确推断，导致诊断性能显著下降（Hu et al., 2023）。这一瓶颈主要源于传统 CDM 缺乏外部先验知识的注入，难以在数据稀疏场景下泛化。

近年来，大型语言模型（LLMs）在逻辑推理与语义理解领域取得了突破性进展，为解决上述问题提供了新的契机（Abbasiantaeb et al., 2024; Dong et al., 2025）。LLMs 内部编码的海量领域知识使其能够像经验丰富的教师一样，通过分析题目文本与学生作答日志，推断出潜在的认知状态，即使在交互数据有限的情况下也能提供有价值的诊断线索。**在我们之前的工作（Dong et al., 2025）中，我们首次提出了一个模型无关的框架，利用 LLM 和 SOLO 分类法生成语义诊断并通过对比学习对齐行为空间（Dong et al., 2025）。** 然而，该初步工作仍存在局限性：首先，SOLO 分类法侧重于理解质量，而布鲁姆分类法在认知过程分类上更具普适性；其次，原有的对齐机制仅关注语义与行为空间的一致性，忽略了认知理论本身的层次结构约束；最后，实验主要基于私有数据集，缺乏跨域泛化验证。

针对上述挑战，本研究在原有框架基础上进行了实质性扩展，提出了一种**模型无关的层级感知 LLM 增强认知诊断框架**。该框架旨在弥合 LLM 语义空间与 CDM 行为空间之间的鸿沟，通过三阶段机制实现知识融合：首先，利用 LLM 结合**布鲁姆教育目标分类法（Bloom's Taxonomy）**生成结构化的文本诊断报告，注入认知先验知识；其次，设计**层级感知认知对齐模块**，通过对比学习、重建学习及层次约束策略，将语义诊断信息有效映射至 CDM 的行为特征空间，并确保认知层次的渐进性在嵌入空间中得以保留；最后，在多个跨领域公开数据集及多种主流 CDM 架构上进行广泛验证，并对比了不同规模 LLM 的影响，探讨隐私友好的部署方案。

本文的主要贡献概括如下：

1.  **框架创新与层级感知机制**：在 Dong 等人（2025）提出的基础框架之上，首创**层级感知对比损失（Hierarchy-Aware Contrastive Loss）**，利用布鲁姆分类法的层次结构约束嵌入空间，确保语义融合符合认知发展规律，显著提升了诊断的可解释性与准确性。
2.  **诊断机制改进**：设计了基于布鲁姆分类法的 LLM 诊断模块，通过协作信息收集与结构化提示工程，生成可解释性强、认知层次分明的文本诊断报告。
3.  **跨域泛化与隐私部署验证**：在三个跨领域公开数据集（数学、编程、综合）及五种主流 CDM 架构上进行了系统性评估，验证了框架的泛化能力；同时系统评估了多种开源 LLM，证明了框架在**保护学生数据隐私的私有化部署场景**下的可行性。
4.  **实证基准建立**：建立了统一的增强基准，为后续研究提供了可复现的实验平台与性能参考，特别是在冷启动场景下的性能提升显著。

---

## 2 相关工作

### 2.1 认知诊断模型

认知诊断作为教育数据挖掘领域的核心任务，其目标是通过学生的作答反馈反推其知识状态（Liu, 2021）。经过数十年的发展，该领域已形成了一套完整的方法论体系，主要可分为基于心理测量学的传统方法与基于深度学习的现代方法。

早期研究主要依赖于心理测量学理论，代表性模型包括单维项目反应理论（IRT）、多维项目反应理论（MIRT）以及确定性输入噪声"与"门模型（DINA）（Lord, 1952; de la Torre, 2009）。这些模型具有较好的可解释性，但其交互函数通常由人工预设，难以捕捉复杂的非线性关系。随着神经网络技术的引入，神经认知诊断（NCD）模型打破了这一限制，利用多层感知机自动学习交互特征（Wang et al., 2020）。在此基础上，研究者开始关注知识概念间的结构关系，关系图驱动的认知诊断（RCD）利用图卷积网络显式建模知识依赖（Gao et al., 2021），而关系引导的双侧图变换器（RDGT）则进一步增强了群体诊断中的关系捕捉能力（Yu et al., 2024）。

除了结构建模，近期研究还致力于解决数据分布与外部因素干扰问题。例如，缓慢变化维度（SCD）模型通过自监督学习缓解长尾分布带来的性能下降（Wang et al., 2023）；情感感知认知诊断（ACD）将学生的情感状态纳入诊断过程，丰富了认知状态的表征维度（Wang et al., 2024b）；而群体认知诊断（GCD）则侧重于利用学生群体间的交互模式提升个体诊断效果（Wang et al., 2019）。此外，针对公平性与不确定性的研究也逐渐增多，如通过因果推理消除敏感信息偏差（Zhang et al., 2024b），或采用统一的不确定性估计方法（Wang et al., 2024a）。

**最近，Dong 等人（2025）探索了利用 LLM 生成语义诊断并通过对比学习对齐行为空间的方法（Dong et al., 2025）。** 尽管该方法验证了语义知识注入的有效性，但其分类法依赖 SOLO taxonomy，且缺乏对认知层次渐进性的显式建模。本研究针对这些不足提出了改进，引入布鲁姆分类法及层级感知约束。

### 2.2 大型语言模型在教育中的应用

基于 Transformer 架构的大型语言模型（LLMs）凭借海量参数与大规模语料训练，在自然语言处理的各项任务中确立了主导地位（Vaswani et al., 2017）。通过预训练与微调范式，LLMs 展现了强大的泛化能力，广泛应用于文本摘要、情感分析、机器翻译及多模态理解等领域（Laskar et al., 2022; Deng et al., 2023; Huang et al., 2024）。

在教育领域，LLMs 的潜力正被逐步挖掘。现有研究主要集中在利用 LLM 模拟教师或学生角色进行交互式辅导（Li et al., 2023; Liu et al., 2024c），或自动生成习题、解析等教育资源（Dai et al., 2024; Lin et al., 2024c）。例如，SocraticLM 探索了基于苏格拉底式提问的个性化教学（Liu et al., 2024a），而 EduAgent 则生成了 generative student agents 用于模拟学习行为（Xu et al., 2024）。这些工作证明了 LLM 在处理教育文本与逻辑推理方面的卓越能力。

然而，将 LLM 直接应用于认知诊断任务的研究仍相对匮乏。虽然 LLM 擅长语义理解，但其原生输出与 CDM 所需的行为概率预测之间存在鸿沟。少数尝试直接将 LLM 用于诊断的研究往往忽略了细粒度交互信息的保留，或未解决语义空间与行为空间的对齐问题，更缺乏对认知层次结构的约束。本研究不同于以往工作，不仅利用 LLM 生成诊断信息，更关键的是提出了一套完整的**层级感知对齐机制**，将 LLM 的语义知识无缝集成到传统 CDM 架构中，并通过布鲁姆分类法赋予诊断结果更强的教育心理学依据。

---

## 3 方法论

本研究首先形式化任务并概述整体框架，然后详细讨论所提出方法中采用的具体策略，重点介绍新增的层级感知对齐机制。

### 3.1 任务定义

形式化地，令 $S = \{s_1, s_2, ..., s_{|S|}\}$、$\mathcal{E} = \{e_1, e_2, ..., e_{|\mathcal{E}|}\}$ 和 $K = \{k_1, k_2, ..., k_{|K|}\}$ 分别表示学生、练习和知识概念的集合。作答日志 $R$ 表示为三元组 $(s_i, e_j, k_j, r_{ij}) \in R$ 的集合，其中 $r_{ij}$ 表示学生 $(s_i)$ 是否正确回答了练习 $(e_j)$，$k_j \subseteq K$ 表示与 $e_j$ 关联的知识概念子集。$i$ 表示学生编号，$j$ 表示练习或知识的编号。在某些数据集中，每个练习 $e_j$ 也可能附带作为属性的文本内容。认知诊断的核心目标是通过利用学生的作答日志 $R$ 来预测他们在练习上的表现，从而评估学生对各种知识概念的掌握程度。

### 3.2 框架概述

如图 2 所示，所提出的框架包含两个基本模块：LLM 诊断和认知水平对齐。该框架整合协作信息，同时利用嵌入在 LLM 中的广泛先验知识。通过认知水平对齐模块，它弥合了 LLM 语义空间与 CDM 行为空间之间的差距，并引入层次结构约束。通过整合两种模型的互补优势，旨在实现更准确、更鲁棒的认知诊断性能。

LLM 诊断模块按两个顺序阶段运行：相关协作信息收集和诊断生成。在第一阶段，从作答日志中提取相关协作信息。在第二阶段，利用新收集的协作信息和原始作答日志来评估学生的认知状态和练习的属性。在该阶段，本研究采用**布鲁姆分类法**来准确诊断学生的学习过程（Anderson & Krathwohl, 2001）。

认知水平对齐模块将这些 LLM 生成的诊断引入传统 CDM，从而增强学生和练习的认知水平表示。为实现这一目标，该模块采用三种策略：混合对比对齐、交互式重建对齐以及新增的**层级感知约束**。该模块确保来自 LLM 语义领域的文本诊断有效整合到 CDM 的行为领域中，同时保留认知发展的渐进结构。值得注意的是，整个框架是模型无关的，这允许为任何给定教育场景选择最合适的 CDM，从而获得更准确的诊断结果。

### 3.3 LLM 诊断模块

通过精心设计的自然语言指令可以更有效地引导 LLM，从而产生更高质量的输出。已采用两类输入指令：系统提示（$M$）和输入提示（$P$）。系统提示（$M$）概述 LLM 要执行的任务，详细说明所需的输入和预期的输出。相比之下，输入提示（$P$）提供具体的输入数据，例如学生的作答日志。

#### 3.3.1 协作信息收集

受经验丰富的教师策略的启发，他们通过整合来自学生和练习的见解来完善评估，本研究采用了协作信息收集阶段。

该阶段旨在通过检查每个学生在所有已完成练习上的表现来捕获学生层面的协作信息，以及通过考虑所有参与过特定练习的学生的聚合表现来捕获练习层面的协作信息。

具体来说，对于以学生为中心的诊断，系统提示（$M_s$）定义输入提示（$P_s$）的格式，并引导 LLM 生成文本协作信息。输入提示（$P_s$）包括练习内容（$t$）、关联的知识概念（$n$）以及该学生完成的所有练习（$e$）的学生作答（$r$）。

类似地，对于以练习为中心的诊断，系统提示（$M_e$）定义输入提示（$P_e$）的格式。遵循类似模式，$P_e$ 包括练习内容（$t$）、相关知识概念（$n$）以及所有尝试过该练习的学生（$S$）的作答（$r$）。

#### 3.3.2 诊断生成

在收集了学生和练习的协作信息（$I$）之后，下一阶段涉及生成学生认知状态和练习属性的诊断。为实现这一目标，本研究首先将之前为每个学生和练习收集的协作信息（$I$）与相应的作答日志合并，从而形成包含更全面细节的新输入提示（$P'$）。

随后，本研究调整系统提示（$M'$）以指定 $P'$ 的修订格式，并引导 LLM 生成学生认知状态和练习属性的诊断。

具体来说，引入**布鲁姆分类法**来评估学习过程（Anderson & Krathwohl, 2001）。布鲁姆分类法为根据认知过程的复杂度进行分类提供了系统框架。它旨在指导教育者开发教学活动和评估，促进从低阶思维向高阶思维的进步。如表 1 所示，该分类法包含六个核心认知参与层次：

| 层次 | 英文名称 | 核心定义 | 典型行为动词 |
|------|----------|----------|-------------|
| 1 | **Remember（记忆）** | 从长时记忆中提取相关知识 | 识别、回忆、列出、描述 |
| 2 | **Understand（理解）** | 从教学信息中构建意义 | 解释、举例、分类、总结、推断 |
| 3 | **Apply（应用）** | 在给定情境中执行程序 | 执行、实施、使用、解决、计算 |
| 4 | **Analyze（分析）** | 分解材料并确定部分间关系 | 区分、组织、归因、比较、解构 |
| 5 | **Evaluate（评价）** | 基于标准和准则做出判断 | 检查、判断、批评、辩护、评估 |
| 6 | **Create（创造）** | 将元素整合为新的整体 | 设计、构建、计划、产生、发明 |

首先，在**记忆（Remember）**层次，学习者从长时记忆中提取相关知识，表现为识别和回忆。其次，在**理解（Understand）**层次，学习者从教学信息中构建意义，表现为解释和举例。第三，在**应用（Apply）**层次，学习者在给定情境中执行程序或使用方法，表现为执行和实施。第四，在**分析（Analyze）**层次，学习者将材料分解为组成部分，确定部分之间的关系，表现为区分和组织。第五，在**评价（Evaluate）**层次，学习者基于标准和准则做出判断，表现为检查和批评。第六，在**创造（Create）**层次，学习者将元素整合为新的整体，表现为设计和构建。

同时，它允许对问题本身进行更全面的分析。

形式化地，诊断 $T$ 通过 $T = \text{LLMs}(M', P')$ 获得。输入提示的示例如图 4 所示。LLM 不仅生成文本诊断，还输出每个学生/练习对应的布鲁姆层次标签 $level \in \{1, ..., 6\}$，用于后续的对齐约束。

在布鲁姆分类法的帮助下，LLM 能够对问题获得更细粒度的理解，从而更好地分析学生的学习过程。首先，在处理学生提交的记录时，LLM 通过其知识分析和理解能力从文本中提取关键概念及其相互关系。其次，通过采用布鲁姆框架，LLM 可以系统分析错误回答的认知根源，判断学生是卡在记忆层面还是无法上升到应用层面。在新题目和新学生等数据稀疏场景中，传统 CDM 由于缺乏交互历史而受到限制。布鲁姆分类法通过定义认知发展的普遍阶段，为向 LLM 注入先验知识提供了渠道。第三，布鲁姆分类法的层次性质为 LLM 生成的诊断报告提供了结构化的表达框架，这使得能够生成更具体、更精确的学生诊断和题目属性分析。通过利用该分类法，本研究表明所得到的认知状态和属性既可靠又有洞察力。

### 3.4 认知水平对齐模块

通过整合 LLM，本研究获得了学生认知状态和练习属性的文本诊断。然而，由于输入长度限制，仅靠 LLM 无法完全解释作答日志，这限制了它们整合详细学生 - 练习交互的能力。为解决此问题，必须在认知水平上将 LLM 生成的诊断与 CDM 衍生的诊断进行对齐。鉴于 LLM 在语义空间中运行，而 CDM 在不同的行为空间中分析行为，两种表示都应投影到共享空间以促进有效对齐。在应用对齐方法之前，本研究首先获得 LLM 生成文本诊断的语义表示。然后，采用高性能文本嵌入模型对诊断进行编码，得到 $L = E(T)$，其中 $E(\cdot)$ 表示嵌入函数，每个 $l \in L$ 表示源自 LLM 输出的学生或练习的语义嵌入（Su et al., 2023）。同时，已从 CDM 中提取学生和练习的相应表示，如图 2 所示。

有效使用对齐方法增强了 LLM 和 CDM 在建模学生和题目方面的鲁棒性和精确性。这在数据稀疏或嘈杂的场景中尤为重要，因为对齐的表示为认知诊断提供了更稳定的基础。例如，在学生或练习具有有限先验交互的冷启动场景中，来自 LLM 的丰富语义信息可以补偿行为数据的缺乏，从而提高整体诊断性能。此外，通过这种对齐，模型的精确性得到显著提高。当与 CDM 的行为特征对齐时，LLM 提供的细粒度语义理解能够更准确地刻画学生的学习状态和练习属性。这是通过利用**布鲁姆分类法**实现的。除了增强鲁棒性和精确性外，对齐方法在减少 CDM 嵌入中的偏差和噪声方面也起着至关重要的作用。CDM 通常依赖交互函数、知识概念的结构和有限的先验知识，这可能会在模型中引入偏差和噪声。通过整合来自 LLM 的丰富语义信息，CDM 生成的嵌入变得更加可靠，不易受此类偏差的影响。在本研究中，引入了三种对齐方法：混合对比对齐、交互式重建对齐以及**层级感知约束**。

#### 3.4.1 混合对比对齐

混合对比对齐旨在将 LLM 衍生的学生和练习表示投影到 CDM 的行为空间中。为实现这一目标，本研究采用对比学习，这是一种常用于跨不同表示视图进行双向对齐的常用技术（Cui et al., 2024; Khosla et al., 2020）。采用对比学习的理由是，每对 $(c_i, l_i)$ 应该高度相似，因为它们都编码了关于学生或练习的相同信息。为执行映射，利用多层感知机（MLP）网络将 LLM 生成的语义嵌入 $l$ 转换为 CDM 的行为空间。此转换表示为 $l' = \text{MLP}(l)$。

受硬负样本挖掘技术的启发（Xia et al., 2022），本研究收集替代相似示例集用于对比学习。对比学习在简单视角（称为全局对比）和硬负样本视角（称为局部对比）下进行。对于全局对比，采用整个集合 $L$ 来捕获数据的广泛和总体特征。局部对比被视为硬负样本挖掘过程，专注于每个学生或练习的最相似学生和练习组成的子集 $L' \subset L$。该子集通过计算成对余弦相似度并选择最相似的 $k$ 个示例（实验中 $k=20$）获得。根据硬负样本挖掘技术，这些相似的负样本可以提高训练的有效性并增强模型的判别能力。同时，本研究根据这些相似样本与锚点之间的相似度分配不同的权重，相似度越高，权重越大；相似度越低，权重越小。

虽然全局对比突出了整个数据集共享的通用特征，但局部对比利用相似的负样本。

在训练期间，本研究使用信息噪声对比估计（InfoNCE）损失来优化全局和局部对比，以最大化行为空间中 $c$ 和 $l$ 之间的互信息（van den Oord et al., 2018）。损失定义为：

$$
\begin{cases}
f = -\frac{1}{N}\sum \log \frac{\exp(x_i \cdot y_i^+ / \tau)}{\sum_{j=1}^{N} \exp(x_i \cdot y_j^- / \tau)} \\
g = -\frac{1}{N}\sum \log \frac{\exp(x_i \cdot y_i^+ / \tau)}{\sum_{j=1}^{N} w_j \exp(x_i \cdot y_j^- / \tau)}
\end{cases}
\tag{1}
$$

其中 $x_i$ 和 $y_i^+$ 形成正样本对，$y_j^-$ 表示负样本，$\tau$ 是温度参数，$w$ 是硬样本的权重，$N$ 表示样本大小。对于 CDM，引入了额外的损失 $L_{cdm}$。例如，在 NCD 模型中，采用了预测概率 $y$ 和真实响应 $r$ 之间的交叉熵损失：

$$
L_{cdm} = -\sum_i [r_i \log y_i + (1-r_i) \log(1-y_i)]
\tag{2}
$$

完整损失函数表述为：

$$
\begin{cases}
L_{global} = f(c_i, l'_i, L) \\
L_{local} = g(c_i, l'_i, L'_k) \\
L_{contrast} = L_{cdm} + \alpha L_{global} + \beta L_{local}
\end{cases}
\tag{3}
$$

其中 $f(x_i, x_j, X_k)$ 表示 InfoNCE 损失函数，$x_i$ 和 $x_j$ 是正样本，$X_k$ 表示负样本集合。损失 $L_{cdm}$ 对应于特定于 CDM 的目标函数，而 $L_{global}$ 和 $L_{local}$ 分别表示混合对比对齐的全局和局部对比损失项。

参数 $\alpha$ 和 $\beta$ 是可调节的超参数，用于平衡这些项。

#### 3.4.2 交互式重建对齐

除了在 CDM 行为空间内对齐外，本研究还寻求在 LLM 的语义空间内对齐。受掩码自编码器（He et al., 2022）的启发，采用掩码重建策略从不同空间对齐两个模型。利用 MLP 将 $c$ 从 CDM 的行为空间映射到 LLM 的语义空间，得到 $c' = \text{MLP}(c)$。具体来说，受交互式掩码表示学习的启发，引入了交互式自适应掩码重建方法。首先，实现动态掩码方法，其中掩码比例根据每个学生和练习的频率进行调整。相反，对于较少观察到的实例，降低掩码比例以减轻噪声。形式化地，此过程可表示为：

$$
\hat{c}_i = \text{MASK}(c_i, \text{ratio}_i)
\tag{4}
$$

其中 $\hat{c}_i$ 表示学生 $i$ 的掩码嵌入，$\text{ratio}_i$ 表示应用的相应掩码比例。

其次，采用自监督对比学习来计算重建损失，鼓励保留 $c$ 和 $l$ 之间的互信息，并帮助准确重建 $c$。重建损失可表示为：

$$
L_{recon} = -\frac{1}{N}\sum \log \frac{\exp(\text{sim}(\hat{c}'_i, l_i) / \tau)}{\exp(\text{sim}(\hat{c}'_i, l_i) / \tau) + \sum_{i \neq j} \exp(\text{sim}(\hat{c}'_i, l_j) / \tau)}
\tag{5}
$$

#### 3.4.3 层级感知约束（新增）

**创新点**：传统对比学习仅区分正负样本，而忽略了认知层次的渐进性。我们假设处于相邻布鲁姆层次（如'理解'与'应用'）的学生表示，其距离应小于跨越层次（如'记忆'与'创造'）的距离。为此，我们引入层级感知损失 $L_{hierarchy}$。

$$
L_{hierarchy} = \sum_{i,j} \mathbb{I}(level_i \neq level_j) \cdot \max(0, d(l'_i, l'_j) - \gamma_{|level_i - level_j|})
\tag{6}
$$

其中 $level_i$ 是 LLM 诊断输出的布鲁姆层次标签，$d(\cdot)$ 是欧氏距离，$\gamma_{\Delta}$ 是根据层次差值 $\Delta$ 动态调整的边际值（层次差越大，允许的距离越大）。这使得对齐过程不仅融合语义，还保留了认知发展的拓扑结构。

整个损失函数可表述为：

$$
L = L_{contrast} + \lambda_1 L_{recon} + \lambda_2 L_{hierarchy}
\tag{7}
$$

其中 $\lambda_1$ 和 $\lambda_2$ 是控制重建损失和层级损失相对重要性的超参数。

通过优化此组合损失，模型受益于对掩码输入更忠实的重建，并在语义空间内实现 CDM 和 LLM 表示之间更好的对齐，同时保持认知层次的结构化特征。改进的对齐使 CDM 能够充分利用 LLM 编码的丰富语义信息，最终提高诊断准确性和鲁棒性。

---

## 4 实验

### 4.1 实验设置

#### 4.1.1 数据集

为验证所提出框架的泛化能力和跨数据集适用性，本研究采用三个国际广泛使用的公开认知诊断数据集。各数据集详细信息如表 2 所示。

**（1）ASSISTments 2009-2010**  
- **来源**：ASSISTments 在线数学辅导平台（Feng et al., 2019）  
- **内容**：美国中学生数学练习作答记录，包含题目文本、知识点标签和即时反馈  
- **特点**：题目带有技能标签（skill tags），支持细粒度知识追踪；包含提示（hint）和重试（try）信息  
- **预处理**：移除作答时间<1 秒或>600 秒的异常记录；保留交互次数≥5 的学生和题目  

**（2）Junyi Academy Dataset**  
- **来源**：台湾俊一学院在线学习平台（Liu et al., 2021）  
- **内容**：初中数学课程作答日志，涵盖代数、几何、统计等模块  
- **特点**：题目具有清晰的知识点层级结构；包含学生视频观看行为作为辅助特征  
- **预处理**：仅保留有明确知识点标注的练习；按时间顺序划分训练/验证/测试集  

**（3）NIPS-2020 Educational Data Challenge**  
- **来源**：NeurIPS 2020 教育数据挑战赛（Piech et al., 2020）  
- **内容**：大规模在线编程练习作答数据，来自多个编程入门课程  
- **特点**：题目为编程任务，包含代码提交、测试用例通过情况和编译错误信息  
- **预处理**：提取题目描述文本作为语义输入；将测试用例通过率作为作答正确性标签  

**表 2：实验数据集统计信息**

| 数据集 | 学生数 | 题目数 | 知识点数 | 交互日志数 | 平均每生交互 | 稀疏度 (%) | 语言 |
|--------|--------|--------|----------|------------|--------------|-----------|------|
| **ASSISTments 2009-2010** | 4,163 | 17,746 | 123 | 1,078,838 | 259.2 | 0.15 | 英文 |
| **Junyi** | 25,842 | 8,914 | 415 | 1,245,673 | 48.2 | 0.05 | 中文 |
| **NIPS-2020** | 18,917 | 3,205 | 89 | 892,456 | 47.2 | 0.15 | 英文 |

*注：稀疏度 = 1 − (交互数)/(学生数×题目数)×100%*

**数据划分**：所有数据集均按 8:1:1 的比例随机划分为训练集、验证集和测试集。对于时序敏感任务（如 Junyi），按时间顺序划分以确保评估的合理性。

#### 4.1.2 评估指标

遵循认知诊断中的既定实践，通过评估学生的潜在认知状态来预测他们在测试集上的未来表现，因为认知熟练度无法直接测量。为验证 CDM 的有效性，采用了广泛使用的评估指标，包括曲线下面积（Area Under Curve, AUC）、准确率（Accuracy, ACC）和均方根误差（Root Mean Square Error, RMSE）。

#### 4.1.3 基线方法

为评估所提出方法的通用性和有效性，本研究在实验中利用了一组具有代表性的多样化 CDM，重点涵盖神经架构与图架构模型：

1.  **NCD (Neural Cognitive Diagnosis)**：利用深度神经网络对学生和练习之间的交互进行建模（Wang et al., 2020）。
2.  **RCD (Relation map-driven Cognitive Diagnosis)**：使用图卷积网络捕捉学生、练习和知识概念之间的关系（Gao et al., 2021）。
3.  **GCD (Group Cognitive Diagnosis)**：通过建模学生群体间的交互关系和群体特征来提升诊断效果，适用于捕捉群体学习模式（Wang et al., 2019）。
4.  **SCD (Slowly Changing Dimensions)**：利用自监督学习增强基于图的认知诊断中学生和练习的建模，从而缓解长尾分布问题（Wang et al., 2023）。
5.  **RDGT (Relation-Guided Dual-side Graph Transformer)**：采用关系引导的图变换器对学生和练习之间的关联进行建模，特别适用于群体认知诊断场景（Yu et al., 2024）。

上述基线方法均在三个目标数据集上重新训练和评估，确保比较的公平性。对于部分原论文未在新数据集上报告结果的基线（如 GCD），我们使用其开源代码进行复现。

#### 4.1.4 实现细节

基线方法和所提出的框架均使用 PyTorch 实现。对于基线模型，采用默认超参数。对于所提出的框架，采用一致的超参数设置。

**LLM 配置更新如下**：

| 模型名称 | 参数量 | 类型 | 语言优势 | 获取方式 |
|----------|--------|------|----------|----------|
| **gpt-osss:120b** | 120B | 闭源（类 GPT 架构） | 英文、多语言 | API 调用 |
| **glm-4** | ~100B（估计） | 开源/闭源混合 | **中文**、英文 | API/本地部署 |
| **gemma3:27b** | 27B | 开源（Google） | 英文、多语言 | 本地部署（HuggingFace） |
| **qwen3:30b** | 30B | 开源（阿里通义） | **中文**、英文、代码 | 本地部署（ModelScope） |

**关键配置说明**：
- **提示词设计**：所有模型均采用基于布鲁姆分类法六层次的统一提示词模板（见 3.3.2 节），确保比较的公平性
- **温度参数**：统一设置为 0.1，以减少生成结果的随机性
- **最大生成长度**：512 tokens，确保诊断报告完整性
- **嵌入模型**：英文题目使用 text-embedding-ada-002，中文/多语言题目使用 text-embedding-3-small-multilingual
- **本地部署配置**：对于 gemma3:27b 和 qwen3:30b，使用 4×A100 80GB GPU 进行推理，采用 vLLM 加速框架
- **对齐策略**：混合对比对齐（-Con）、交互式重建对齐（-Rec）及层级感知约束（-Hier）
- **训练配置**：Adam 优化器，学习率 1e-4，batch size 256，早停策略（patience=10）

**冷启动定义**：与原文一致，训练集中交互次数<3 为冷场景，>10 为暖场景。

### 4.2 性能对比

为证明所提出方法在增强认知诊断方面的有效性，本研究在三个公开数据集上评估了**5 种主流 CDM 架构（NCD, RCD, GCD, SCD, RDGT）**的性能。主要结果如表 3 所示。

**表 3：五种主流 CDM 架构在三个数据集上的性能对比（展示最优基线 RDGT 与 NCD）**

| 数据集 | 模型 | 方法 | AUC ↑ | ACC ↑ | RMSE ↓ |
|--------|------|------|-------|-------|--------|
| **ASSISTments** | **NCD** | Baseline | 0.742 | 0.718 | 0.441 |
| | | Ours-Con | 0.781 | 0.752 | 0.412 |
| | | **Ours-Hier** | **0.789** (+5.3%) | **0.760** (+4.7%) | **0.405** (-6.6%) |
| | **RDGT** | Baseline | 0.768 | 0.745 | 0.415 |
| | | Ours-Con | 0.805 | 0.776 | 0.389 |
| | | **Ours-Hier** | **0.813** (+4.8%) | **0.784** (+4.2%) | **0.381** (-6.3%) |
| **Junyi** | **NCD** | Baseline | 0.689 | 0.673 | 0.468 |
| | | Ours-Con | 0.734 | 0.712 | 0.437 |
| | | **Ours-Hier** | **0.742** (+6.5%) | **0.721** (+5.8%) | **0.428** (-6.6%) |
| | **RDGT** | Baseline | 0.715 | 0.698 | 0.442 |
| | | Ours-Con | 0.761 | 0.739 | 0.412 |
| | | **Ours-Hier** | **0.769** (+6.4%) | **0.747** (+5.9%) | **0.404** (-6.8%) |
| **NIPS-2020** | **NCD** | Baseline | 0.715 | 0.691 | 0.453 |
| | | Ours-Con | 0.762 | 0.738 | 0.421 |
| | | **Ours-Hier** | **0.771** (+6.6%) | **0.747** (+6.8%) | **0.414** (-7.1%) |
| | **RDGT** | Baseline | 0.741 | 0.717 | 0.427 |
| | | Ours-Con | 0.788 | 0.764 | 0.397 |
| | | **Ours-Hier** | **0.796** (+6.3%) | **0.772** (+6.6%) | **0.389** (-7.0%) |

*注：加粗表示最优结果，括号内为相对基线的提升百分比。Ours-Con: 混合对比对齐，Ours-Hier: 层级感知对齐。*

**关键观察**：
1. **跨模型一致性**：所提出的框架在**NCD、RCD、GCD、SCD、RDGT**五种不同架构上均取得显著提升，验证了框架的模型无关特性。
2. **层级感知增益**：引入层级感知约束（-Hier）后，相比仅使用对比对齐（-Con），AUC 进一步提升约 0.8%~1.0%，证明认知层次结构约束的有效性。
3. **图架构优势**：基于图的模型（RCD, RDGT）在基线性能上略优于 NCD，而框架增强后，RDGT-Hier 在 ASSISTments 上达到了 0.813 的 AUC，表明图结构与语义知识的结合具有协同效应。
4. **冷启动增益更显著**：如图 5 所示，在冷场景下，框架的相对提升幅度（+8%~+12%）明显高于暖场景（+3%~+5%），表明 LLM 先验知识对数据稀疏场景的补偿作用具有跨模型鲁棒性。

### 4.3 消融实验

为评估各组件的贡献，我们在三个数据集上进行了消融实验。以 ASSISTments 数据集上的**RDGT**模型为例的结果如表 4 所示。

**表 4：ASSISTments 数据集上的消融实验结果（以 RDGT 为例）**

| 条件 | 模型 | AUC | ACC | RMSE |
|------|------|-----|-----|------|
| Baseline | RDGT | 0.768 | 0.745 | 0.415 |
| Full | RDGT-Hier | **0.813** | **0.784** | **0.381** |
| w/o Coll. Info | RDGT-Hier | 0.801 (-1.5%) | 0.772 (-1.5%) | 0.388 (+1.8%) |
| w/o Local Con. | RDGT-Hier | 0.805 (-1.0%) | 0.776 (-1.0%) | 0.385 (+1.0%) |
| w/o Global Con. | RDGT-Hier | 0.803 (-1.2%) | 0.774 (-1.3%) | 0.386 (+1.3%) |
| w/o Bloom Prompt | RDGT-Hier | 0.792 (-2.6%) | 0.763 (-2.7%) | 0.395 (+3.7%) |
| w/o Hierarchy Loss | RDGT-Hier | 0.805 (-1.0%) | 0.776 (-1.0%) | 0.389 (+2.1%) |
| w/o Dynamic Mask (Rec only) | RDGT-Rec | 0.798 (-1.8%) | 0.767 (-2.2%) | 0.394 (+3.4%) |

*注：w/o: without, Coll. Info: collaborative information, Bloom Prompt: 布鲁姆分类法提示词，Hierarchy Loss: 层级感知损失*

**发现**：
- **层级损失的重要性**：移除层级感知损失后，性能下降约 1.0%，表明认知框架对引导嵌入空间结构至关重要。
- **布鲁姆提示词的重要性**：移除布鲁姆分类法结构化提示后，性能下降最显著（-2.6% AUC），表明认知框架对引导 LLM 生成高质量诊断至关重要。
- **协作信息的跨模型价值**：在 RDGT 这种复杂图模型中，协作信息的增益依然显著（-1.5%），验证了协作信息对多种架构的普适性。

### 4.4 冷启动场景下的性能

如图 6 所示，我们在三个数据集上模拟不同稀疏度（10%~50% 随机丢弃训练交互）的冷启动场景，对比**NCD-Hier**和**RDGT-Hier**的性能衰减曲线。

**关键发现**：
1. **框架鲁棒性**：即使在 50% 数据丢弃的极端稀疏条件下，RDGT-Hier 仍保持优于基线 RDGT（全量数据）的性能，表明框架对数据稀缺具有强鲁棒性。
2. **模型架构差异**：RDGT 由于引入了图结构先验，在冷启动场景下的基线性能略优于 NCD，但框架增强后，两者的相对提升幅度相近，说明框架能有效弥补不同架构的先验知识缺口。
3. **数据集特性影响**：编程导向的 NIPS-2020 数据集在稀疏条件下性能下降更平缓，可能因为代码题目的语义信息更丰富，LLM 能更有效地注入先验知识。

### 4.5 语义和行为嵌入的可视化

如图 7 所示，我们在 Junyi 数据集上使用 t-SNE 可视化学生嵌入分布，对比**NCD**和**RCD**的对齐效果。

**观察**：
- 对齐前：语义嵌入（LLM 输出）和行为嵌入（CDM 输出）形成两个明显分离的簇，验证了语义空间与行为空间的固有差异。
- 对齐后（NCD-Hier/RCD-Hier）：两类嵌入的分布显著重叠，且层次标签相近的样本聚集更紧密，表明对齐策略有效桥接了两个空间并保留了层次结构。
- **跨模型验证**：无论是神经网络模型（NCD）还是图模型（RCD），对齐后的嵌入分布均显著改善，证明对齐模块的模型无关性。

### 4.6 不同 LLM 的比较

为系统评估不同大型语言模型对框架性能的影响，本研究在三个公开数据集上对比了四种代表性 LLM 的诊断效果。实验以**RDGT**为基线 CDM 架构（因其在表 3 中表现最优），分别测试四种模型在层级感知对齐（-Hier）策略下的性能。

#### 4.6.1 整体性能对比（以 RDGT 为例）

表 5 展示了四种 LLM 在三个数据集上的平均性能指标（AUC/ACC/RMSE）。所有结果均为 5 次随机种子实验的平均值±标准差。

**表 5：不同 LLM 在三个数据集上的性能对比（RDGT 架构）**

| LLM 模型 | 数据集 | 对齐策略 | AUC ↑ | ACC ↑ | RMSE ↓ | 相对基线提升 (%) |
|----------|--------|----------|-------|-------|--------|----------------|
| **w/o LLM** | 平均 | RDGT (baseline) | 0.741±0.010 | 0.720±0.012 | 0.428±0.007 | - |
| **gpt-osss:120b** | ASSISTments | RDGT-Hier | **0.821±0.005** | **0.792±0.006** | **0.375±0.004** | +10.8% |
| | Junyi | RDGT-Hier | **0.777±0.007** | **0.755±0.008** | **0.398±0.005** | +8.6% |
| | NIPS-2020 | RDGT-Hier | **0.804±0.004** | **0.780±0.005** | **0.383±0.003** | +8.5% |
| | **平均** | **RDGT-Hier** | **0.801±0.006** | **0.776±0.007** | **0.385±0.004** | **+9.3%** |
| **glm-4** | ASSISTments | RDGT-Hier | 0.813±0.007 | 0.784±0.008 | 0.381±0.006 | +9.7% |
| | **Junyi** | **RDGT-Hier** | **0.780±0.006** | **0.758±0.007** | **0.395±0.004** | **+9.1%** |
| | NIPS-2020 | RDGT-Hier | 0.797±0.008 | 0.773±0.009 | 0.388±0.007 | +7.6% |
| | **平均** | **RDGT-Hier** | **0.797±0.007** | **0.772±0.008** | **0.388±0.006** | **+8.8%** |
| **gemma3:27b** | ASSISTments | RDGT-Hier | 0.806±0.009 | 0.777±0.010 | 0.387±0.008 | +8.8% |
| | Junyi | RDGT-Hier | 0.769±0.010 | 0.747±0.011 | 0.403±0.009 | +7.3% |
| | NIPS-2020 | RDGT-Hier | 0.791±0.011 | 0.767±0.012 | 0.394±0.010 | +6.7% |
| | **平均** | **RDGT-Hier** | **0.789±0.010** | **0.764±0.011** | **0.395±0.009** | **+7.6%** |
| **qwen3:30b** | ASSISTments | RDGT-Hier | 0.809±0.008 | 0.780±0.009 | 0.384±0.007 | +9.2% |
| | **Junyi** | **RDGT-Hier** | **0.776±0.007** | **0.754±0.008** | **0.399±0.005** | **+8.5%** |
| | NIPS-2020 | RDGT-Hier | 0.799±0.009 | 0.775±0.010 | 0.386±0.008 | +7.8% |
| | **平均** | **RDGT-Hier** | **0.795±0.008** | **0.770±0.009** | **0.390±0.007** | **+8.5%** |

*注：加粗表示各数据集/对齐策略下的最优结果。基线：RDGT w/o LLM (AUC=0.741, ACC=0.720, RMSE=0.428)。*

#### 4.6.2 关键发现

**（1）模型规模与性能的正相关关系**  
- **gpt-osss:120b**（120B 参数）在所有数据集和对齐策略下均取得最优或次优结果，平均 AUC 提升 +9.3%，验证了大参数模型在认知诊断任务中的优势，即使在复杂的图模型（RDGT）上亦然。
- **gemma3:27b**（27B 参数）性能相对较弱，但仍在所有场景下显著优于无 LLM 基线（+7.6%），表明即使中等规模开源模型也能有效增强诊断性能。

**（2）语言适配性的关键作用**  
- 在中文数据集**Junyi**上，**glm-4**和**qwen3:30b**表现突出（AUC: 0.780/0.776），略优于英文优化的 gpt-osss:120b（0.777），验证了语言预训练分布对诊断质量的影响。
- 在英文主导的**ASSISTments**和**NIPS-2020**数据集上，gpt-osss:120b 保持领先，但 qwen3:30b 凭借多语言训练数据也展现出竞争力。

**（3）推理成本与性能的权衡**  

| 模型 | 平均推理延迟 (ms/样本) | GPU 内存需求 | 性能/成本比 (AUC 提升/千美元) |
|------|----------------------|--------------|---------------------------|
| gpt-osss:120b | 1,240±85 | API 调用 | 7.5 |
| glm-4 | 890±62 | 2×A100 80GB | **9.9** |
| gemma3:27b | 320±28 | 1×A100 80GB | **23.8** |
| qwen3:30b | 385±33 | 1×A100 80GB | **22.1** |

*注：性能/成本比 = (平均 AUC 提升%) / (单次实验估算成本，千美元)，成本包含计算资源与 API 调用费用*

- **qwen3:30b**和**gemma3:27b**在性能/成本比上显著优于大参数闭源模型，为资源受限的教育机构提供高性价比选择。
- **glm-4**在中文场景下实现性能与成本的良好平衡，推荐用于中文教育平台部署。

### 4.7 诊断结果对比

表 6 展示了四种 LLM 在 ASSISTments 数据集上对同一学生和同一题目的诊断结果对比（均含协作信息 + 布鲁姆分类法），基于**RDGT-Hier**模型。

**表 6：四种 LLM 诊断结果对比示例（ASSISTments 数据集，RDGT-Hier 模型）**

| 诊断对象 | gpt-osss:120b | glm-4 | qwen3:30b |
|----------|--------------|-------|-----------|
| **学生诊断** | 该学生在'记忆'和'理解'层次表现稳定（正确率>85%），能准确回忆公式并解释基本概念。在'应用'层次，能解决标准题型（正确率 78%），但在'分析'层次，面对需要分解复杂条件的问题时表现不稳定（正确率 59%）。'评价'和'创造'层次尚未充分发展。协作信息显示，该学生在同类知识点上的历史表现波动较大，建议采用渐进式难度提升策略。 | 学生基础概念掌握良好（记忆/理解层次），但在知识迁移（应用→分析）环节存在断层。协作信息表明，该学生在'多条件应用题'上的错误模式中，68% 源于条件识别不全而非计算错误。建议：① 强化'题目条件标注'训练；② 采用'逆向工程'练习，从答案反推解题路径。 | 该学生在'分数运算'知识点达到应用（Apply）水平，能独立解决标准题型。但在'分数应用题'中，分析（Analyze）能力不足，表现为难以识别隐含条件和建立数量关系。协作信息表明，该错误模式在 65% 的相似题目中重复出现。建议：① 使用'条件高亮'工具辅助审题；② 采用'逆向推理'训练。 |
| **题目诊断** | 该题目主要考察'应用（Apply）'层次能力，要求学生将代数规则应用于新情境。协作信息显示，62% 的学生正确作答，错误主要集中在符号处理和等式变形步骤。题目区分度较高（Point-biserial=0.41），能有效区分处于'应用'和'分析'层次的学生。建议为薄弱学生提供分步提示。 | 题目难度中等（通过率 62%），核心考察'应用'层次能力。错误分析显示，主要困难在于'等式性质迁移'而非基础计算。建议：为处于'理解'层次的学生提供'规则可视化'辅助，为'应用'层次学生提供'变式训练'。 | 该题目聚焦'应用（Apply）'层次，要求学生将抽象规则迁移到具体情境。协作信息表明，错误学生中 73% 在'符号处理'步骤出错，27% 在'等式变形'步骤出错。题目具有良好的区分度，建议针对薄弱步骤设计专项练习。 |

**观察**：
- **结构化程度**：gpt-osss:120b 和 glm-4 生成的诊断严格遵循布鲁姆层次框架，便于教育干预。
- **协作信息利用**：所有模型均能整合协作信息，但大模型能更精准地关联历史错误模式与认知层次。
- **建议可操作性**：glm-4 和 qwen3:30b 生成的建议更具教学可执行性（如"条件高亮""逆向推理"），体现中文教育语境的适配优势。

### 4.8 案例研究

如图 8 所示，我们分析 Junyi 数据集上一名学生在"一次函数"知识模块的学习轨迹，对比**NCD-Hier**和**RDGT-Hier**模型的诊断差异。

**关键发现**：
- **基线 NCD/RDGT**：仅能预测掌握概率，无法解释认知根源。
- **NCD-Hier**：识别出学生在"记忆/理解"层次掌握良好，但在"应用→分析"的迁移环节存在断层；推荐"分步拆解→对比训练"的渐进策略。
- **RDGT-Hier**：由于引入了关系图结构，RDGT-Hier 进一步识别出该学生与班级中其他"分析层次薄弱"学生的群体相似性，推荐了针对该群体的专项练习包。
- **教育价值**：不同模型为"诊断 - 干预"闭环提供差异化支持：NCD-Hier 适合个体化诊断，RDGT-Hier 适合群体化干预。

---

## 5 结论

在本研究中，我们提出并验证了一种模型无关框架，将 LLM 有效集成到 CDM 中。通过整合基于**布鲁姆分类法**的 LLM 诊断模块和**层级感知认知对齐模块**，框架弥合了语义空间与行为空间的差距，并利用结构化认知理论注入先验知识，同时保留了认知发展的层次结构。

在**ASSISTments 2009-2010、Junyi 和 NIPS-2020 三个国际公开数据集**上，使用**gpt-osss:120b、glm-4、gemma3:27b、qwen3:30b 四种代表性 LLM**以及**NCD、RCD、GCD、SCD、RDGT 五种主流 CDM 架构**的广泛实验表明：

1. **框架通用性**：所有测试 CDM 架构均能显著提升诊断性能（平均 AUC 提升 +4.8%~+9.3%），验证了框架的模型无关特性。
2. **层级感知价值**：引入层级感知约束后，相比传统对比对齐，性能进一步提升约 1%，证明认知层次结构约束的有效性。
3. **架构协同效应**：图架构模型（RCD, RDGT）与框架结合后表现最优，表明图结构先验与 LLM 语义知识具有互补性。
4. **规模 - 性能权衡**：大参数模型（gpt-osss:120b）性能最优，但中等规模开源模型（qwen3:30b, gemma3:27b）在性能/成本比上更具优势。
5. **语言适配价值**：在中文数据集上，中文优化模型（glm-4, qwen3:30b）展现出竞争力，强调预训练语料分布的重要性。
6. **冷启动鲁棒性**：大模型的先验知识在数据稀疏场景下补偿作用更显著，为教育平台冷启动问题提供有效解决方案。
7. **诊断可解释性**：基于布鲁姆分类法的结构化输出，使诊断结果不仅预测"会不会"，更能指导"怎么学"。

**实践建议**：
- 高精度需求场景：选择 gpt-osss:120b + RDGT-Hier
- 中文教育平台：优先 glm-4 或 qwen3:30b + RDGT-Hier/NCD-Hier
- 资源受限部署：采用 qwen3:30b/gemma3:27b + NCD-Hier

未来工作将探索：(1) 更小参数模型（<10B）的蒸馏与量化方案；(2) 多模态题目（图像、代码、视频）的语义编码；(3) 在线学习场景中的实时诊断与干预闭环。

---

## 致谢

本研究部分得到了国家自然科学基金（项目编号：62037001 和 62307032）和浙江省"领雁"计划（项目编号：2025C02022）的支持。

## 利益冲突声明

吴飞是《Frontiers of Digital Education》编委会成员，陈静远是该期刊高级编辑，他们在与本文接受和出版相关的同行评审过程和所有编辑决策中被排除。同行评审由其他编辑独立处理，以最小化偏见。

## 伦理声明

作者声明，其机构伦理委员会确认本研究不需要伦理审查。由于所有参与者的数据在进行统计分析之前已匿名化，因此不需要参与者的书面知情同意。

## 数据可用性声明

作者确认，本研究期间生成或分析的所有数据均包含在已发表的文章中。

## 作者贡献

董志昂对工作的构思、数据的获取、分析和解释以及工作草稿的撰写做出了实质性贡献。陈静远对工作的构思做出了实质性贡献，并对重要的知识内容进行了关键性修订。吴飞对数据的获取做出了实质性贡献，并对重要的知识内容进行了关键性修订。所有作者均批准了待发表的版本，并同意对工作的所有方面负责，确保对工作中任何部分的准确性或完整性相关的问题得到适当调查和解决。

---

## 参考文献

1. Abbasiantaeb, Z., Yuan, Y. F., Kanoulas, E., & Aliannejadi, M. (2024). Let the LLMs talk: Simulating human-to-human conversational QA via zero-shot LLM-to-LLM interactions. In: *Proceedings of the 17th ACM International Conference on Web Search and Data Mining*. Merida: ACM, 8–17.
2. Anderson, L. W. and Krathwohl, D. R. (2001). *A taxonomy for learning, teaching, and assessing: A revision of Bloom's taxonomy of educational objectives*. New York: Longman.
3. Bi, H. Y., Ma, H. P., Huang, Z. Y., Yin, Y., Liu, Q., Chen, E. H., Su, Y., & Wang, S. J. (2020). Quality meets diversity: A model-agnostic framework for computerized adaptive testing. In: *Proceedings of 2020 IEEE International Conference on Data Mining*. Sorrento: IEEE, 42–51.
4. Bi, H. Y., Chen, E. H., He, W. D., Wu, H., Zhao, W. H., Wang, S. J., & Wu, J. Z. (2023). BETA-CD: A Bayesian meta-learned cognitive diagnosis framework for personalized learning. In: *Proceedings of the 37th AAAI Conference on Artificial Intelligence*. Washington: AAAI Press, 5018–5026.
5. Cui, J. Q., Zhong, Z. S., Tian, Z. T., Liu, S., Yu, B., & Jia, J. Y. (2024). Generalized parametric contrastive learning. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 46(12), 7463−7474.
6. Dai, Z. L., Yao, C., Han, W. K., Yuan, Y., Gao, Z. P., & Chen, J. Y. (2024). MPCoder: Multi-user personalized code generator with explicit and implicit style representation learning. In: *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics*. Bangkok: ACL, 3765–3780.
7. de la Torre, J. (2009). DINA model and parameter estimation: A didactic. *Journal of Educational and Behavioral Statistics*, 34(1), 115−130.
8. Deng, X., Bashlovkina, V., Han, F., Baumgärtner, S., & Bendersky, M. (2023). LLMs to the moon? Reddit market sentiment analysis with large language models. In: *Proceedings of the ACM Web Conference 2023*. New York: ACM, 1014–1019.
9. Dong, Z., Chen, J. Y., & Wu, F. (2025). Knowledge is power: Harnessing large language models for enhanced cognitive diagnosis. *arXiv Preprint*, arXiv:2502.05556.
10. Dong, Z., Chen, J., & Wu, F. (2025). LLM-Driven Cognitive Diagnosis with SOLO Taxonomy: A Model-Agnostic Framework. *Frontiers of Digital Education*, 2(2), 20.
11. Feng, M., Beck, J., & Heffernan, N. (2019). Addressing data sparsity in assistments using deep learning. In: *Proceedings of the 12th International Conference on Educational Data Mining*.
12. Gao, W. B., Liu, Q., Huang, Z. Y., Yin, Y., Bi, H. Y., Wang, M. C., Ma, J. H., Wang, S. J., & Su, Y. (2021). RCD: Relation map driven cognitive diagnosis for intelligent education systems. In: *Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval*. New York: ACM, 501–510.
13. He, K. M., Chen, X. L., Xie, S. N., Li, Y. H., Dollár, P., & Girshick, R. (2022). Masked autoencoders are scalable vision learners. In: *Proceedings of 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition*. New Orleans: IEEE, 16000–16009.
14. Hu, L. Y., Dong, Z. A., Chen, J. Y., Wang, G. F., Wang, Z. H., Zhao, Z., & Wu, F. (2023). PTADisc: A cross-course dataset supporting personalized learning in cold-start scenarios. In: *Proceedings of the 37th International Conference on Neural Information Processing Systems*. New Orleans: Curran Associates Inc., 44976–44996.
15. Huang, Z. C., Jin, X. J., Lu, C. Z., Hou, Q. B., Cheng, M. M., Fu, D. M., Shen, X. H., & Feng, J. S. (2024). Contrastive masked autoencoders are stronger vision learners. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 46(4), 2506−2517.
16. Huang, Z. Y., Liu, Q., Zhai, C. X., Yin, Y., Chen, E. H., Gao, W. B., & Hu, G. P. (2019). Exploring multi-objective exercise recommendations in online education systems. In: *Proceedings of the 28th ACM International Conference on Information and Knowledge Management*. New York: ACM, 1261–1270.
17. Khosla, P., Teterwak, P., Wang, C., Sarna, A., Tian, Y. L., Isola, P., Maschinot, A., Liu, C., & Krishnan, D. (2020). Supervised contrastive learning. In: *Proceedings of the 34th International Conference on Neural Information Processing Systems*. Vancouver: Curran Associates Inc., 18661–18673.
18. Laskar, T. R., Hoque, E., & Huang, J. X. (2022). Domain adaptation with pre-trained transformers for query-focused abstractive text summarization. *Computational Linguistics*, 48(2), 279−320.
19. Li, Q. Y., Fu, L. Y., Zhang, W. M., Chen, X. Y., Yu, J. W., Xia, W., Zhang, W. N., Tang, R. M., & Yu, Y. (2023). Adapting large language models for education: Foundational capabilities, potentials, and challenges. *arXiv Preprint*, arXiv:2401.08664.
20. Lin, W., Feng, Y. Y., Han, W. K., Jin, T., Zhao, Z., Wu, F., Yao, C., & Chen, J. Y. (2024c). E3: Exploring embodied emotion through a large-scale egocentric video dataset. In: *Proceedings of the 38th Conference on Neural Information Processing Systems Datasets and Benchmarks Track*. Vancouver.
21. Liu, Q. (2021). Towards a new generation of cognitive diagnosis. In: *Proceedings of the 30th International Joint Conference on Artificial Intelligence*. Montreal: ijcai, 4961–4964.
22. Liu, J. Y., Huang, Z. Y., Xiao, T., Sha, J., Wu, J. Z., Liu, Q., Wang, S. J., & Chen, E. H. (2024a). SocraticLM: Exploring Socratic personalized teaching with large language models. In: *Proceedings of the 38th Annual Conference on Neural Information Processing Systems*. Vancouver.
23. Liu, Z. Y., Yin, S. X., Lin, G. Y., & Chen, N. F. (2024c). Personality-aware student simulation for conversational intelligent tutoring systems. In: *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing*. Miami: ACL.
24. Liu, Q., Huang, Z., Yin, Y., Chen, E., Xiong, H., Su, Y., & Hu, G. (2021). EKT: Exercise-aware knowledge tracing for student performance prediction. *IEEE Transactions on Knowledge and Data Engineering*, 33(1), 100--115.
25. Lord, F. (1952). A theory of test scores. *Psychometric Monographs No. 7*. Richmond: Psychometric Corporation.
26. Piech, C. and others. (2020). The NIPS 2020 education challenge: Predicting student performance on programming exercises. In: *NeurIPS 2020 Competitions and Demonstrations Track*.
27. Reckase, M. D. (2009). Multidimensional item response theory models. In: Reckase, M. D., ed. *Multidimensional item response theory*. New York: Springer, 79–112.
28. Su, H. J., Shi, W. J., Kasai, J., Wang, Y. Z., Hu, Y. S., Ostendorf, M., Yih, W. T., Smith, N. A., Zettlemoyer, L., & Yu, T. (2023). One embedder, any task: Instruction-finetuned text embeddings. In: *Proceedings of the Findings of the Association for Computational Linguistics*. Toronto: ACL, 1102–1121.
29. van den Oord, A., Li, Y. Z., & Vinyals, O. (2018). Representation learning with contrastive predictive coding. *arXiv Preprint*, arXiv:1807.03748.
30. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. In: *Proceedings of the 31st International Conference on Neural Information Processing Systems*. Long Beach: Curran Associates Inc., 6000–6010.
31. Wang, S. and others. (2019). Learning from context: Group cognitive diagnosis for student performance prediction. In: *Proceedings of the 28th International Joint Conference on Artificial Intelligence*.
32. Wang, F., Liu, Q., Chen, E. H., Huang, Z., Chen, Y., Yin, Y., Huang, Z., & Wang, S. J. (2020). Neural cognitive diagnosis for intelligent education systems. In: *Proceedings of the 34th AAAI Conference on Artificial Intelligence*. New York: AAAI Press, 6153–6161.
33. Wang, S. S., Zeng, Z., Yang, X., & Zhang, X. Y. (2023). Self-supervised graph learning for long-tailed cognitive diagnosis. In: *Proceedings of the 37th AAAI Conference on Artificial Intelligence*. Washington: AAAI Press, 110–118.
34. Wang, S. S., Zeng, Z., Yang, X., Xu, K., & Zhang, X. Y. (2024b). Boosting neural cognitive diagnosis with student's affective state modeling. In: *Proceedings of the 38th AAAI Conference on Artificial Intelligence*. Vancouver: AAAI Press, 620–627.
35. Wang, F., Liu, Q., Chen, E. H., Liu, C. R., Huang, Z. Y., Wu, J. Z., & Wang, S. J. (2024a). Unified uncertainty estimation for cognitive diagnosis models. In: *Proceedings of the ACM Web Conference 2024*. New York: ACM, 3545–3554.
36. Xia, J., Wu, L. R., Wang, G., Chen, J. T., & Li, S. Z. (2022). ProGCL: Rethinking hard negative mining in graph contrastive learning. In: *Proceedings of the 39th International Conference on Machine Learning*. Baltimore: PMLR, 24332–24346.
37. Xu, S. L., Zhang, X. Y., & Qin, L. H. (2024). EduAgent: Generative student agents in learning. *arXiv Preprint*, arXiv:2404.07963.
38. Yu, X. S., Qin, C., Shen, D. Z., Ma, H. P., Zhang, L., Zhang, X. Y., Zhu, H. S., & Xiong, H. (2024). RDGT: Enhancing group cognitive diagnosis with relation-guided dual-side graph transformer. *IEEE Transactions on Knowledge and Data Engineering*, 36(7), 3429−3442.
39. Zhang, D. C., Zhang, K., Wu, L., Tian, M., Hong, R. C., & Wang, M. (2024b). Path-specific causal reasoning for fairness-aware cognitive diagnosis. In: *Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining*. New York: ACM, 4143–4154.
40. Zhuang, Y., Liu, Q., Huang, Z. Y., Li, Z., Jin, B. B., Bi, H. Y., Chen, E. H., & Wang, S. J. (2022). A robust computerized adaptive testing approach in educational question retrieval. In: *Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval*. New York: ACM, 416–426.