# ICDE2025 Response Letter

**Paper ID**

837

**Paper Title**

FedEcover: Fast and Stable Converging Model-Heterogeneous Federated Learning with Efficient-Coverage Submodel Extraction

**Track Name**

RESEARCH PAPER SECOND SUBMISSION CYCLE

---

Dear Program Committee Chairs and Reviewers,

Thank you for your valuable feedback and giving an opportunity to improve our manuscript. We acknowledge the importance of reviews for promoting our work and have carefully addressed all comments and suggestions from the reviewers. Below, we provide a point-by-point response to the reviewers’ concerns. The revised manuscript highlights major changes in ***red text*** for easy reference.

## Response to Reviewer #1

### Response to comment O1-1

>O1-Novelty: The proposed submodel extraction scheme, while intuitive, appears relatively straightforward. It would be beneficial to include additional toy experiments highlighting how random sampling fails to cover all neurons (or important parameters) compared to the proposed approach. Such demonstrations would clarify and strengthen the significance of this solution.

Thank you for highlighting the importance of clarifying the novelty and empirical advantages of our submodel extraction scheme. We agree that demonstrating its superiority over naive random sampling is critical.

As suggested, we conducted a dedicated experiment to compare neuron selection patterns across submodel extraction schemes (Section V-B, Fig. 4). Using a fully-connected layer with 2048 neurons as example (from an actual run of our 3237-client experiment on FEMNIST), we explicitly demonstrates that ***naive randomness fails to guarantee parameter space coverage efficiency (considering covering all space with equal training frequencies), while FedEcover’s buffer-based cyclic sampling systematically eliminates the unevenness***, promoting convergence rate and final accuracy performance for evenly training all parameters on distributed data.

![alt text](images/image.png)

- Static (Fig. 4a) extraction creates severe polarization--25% of neurons trained more frequently while 50% of neurons trained less--leaving high-index neurons untrained and low-index neurons over-trained.
- Rolling (Fig. 4b) scheme demonstrates periodic deviation patterns--32.57% of neurons trained more frequently while 50.63% of neurons trained less--adjacent neurons share synchronized selection frequencies, creating localized overfitting regions (peaks) and undertrained valleys.
- Naive random (Fig. 4c) extraction introduces uneven coverage gaps due to memoryless sampling, violating ergodicity--31.05% of neurons trained more frequently while 30.37% of neurons trained less--some neurons are neglected due to memoryless sampling, while others are redundantly selected.
- FedEcover (Fig. 4d) achieves near-uniform coverage (100% deviations bounded within $\pm 1$), ensuring all neurons are trained evenly.

**Novelty perspective**: FedEcover introduces two key innovations that fundamentally address critical limitations in federated submodel extraction. Besides the even neuron selection we discussed above, FedEcover also introduces structure-aware extraction considerding residual connections.

Residual connections constitute essential computation pathways for modern architectures (e.g., in ResNet, Transformer). Prior work like Federated Dropout [1] employs arbitrary indexing that catastrophically disrupts these structural patterns. This creates *parameter-level semantic mismatch* during aggregation, ultimately corrupting model semantics. For example, in Fig. 3(c), one submodel trains residual connections: $\{b1+a3, d1+b3, f1+e3\}$ on local data but the actual residual connections aggregated on the server is $\{a1+a3, b1+b3, \cdots, f1+f3\}$, which is not aligned with the locally trained ones. This causes invalid training, i.e., the performance of the global model obtained through such aggregation does not improve with rounds, because the semantics of the aggregated parameters are incorrect.

![alt text](images/image-2.png)

FedEcover solves this by enforcing *structural consistency constraints*--maintaining identical neuron indices for connected layers. This innovation enables first-of-its-kind support for complex architectures in submodel extraction and its insight is possible to extend to more architectures requiring structural consistency.

### Response to comment O1-2

>Moreover, employing a step-size or learning rate decay in FL is not entirely new and might be perceived as an incremental improvement. Providing more theoretical insight or deeper empirical comparisons could help underscore the value of this technique within the framework.

We sincerely appreciate this insightful comment. While step-size decay is indeed a classic technique in optimization, its application in model-heterogeneous FL addresses unique challenges that existing works have not systematically studied. We emphasize that the novelty of our global aggregation step-size decay (GSD) lies in its explicit role in stabilizing convergence under dual heterogeneity (data and model, especially with the extra model heterogeneity layer).

In model-heterogeneous FL, an additional layer of complexity arises from structural heterogeneity: parameters receive updates from varying subsets of clients due to submodel extraction. This creates non-stationary update distributions, amplifying convergence instability. Unlike client's *local learning rate decay or local steps decay*[4], GSD is specifically designed to mitigate oscillations through controling the global aggregation step-size. In early stages, a larger step-size ($\eta_g \approx 1$) accelerates knowledge integration from diverse submodels; while in later stages, decaying $\eta_g$ suppresses noise from biased updates caused by sparse parameter participation (e.g., parameters updated only by limited specific clients).

To isolate the impact of model heterogeneity (independent of data heterogeneity), we conducted experiments under IID data distributions (Section VI-E, Fig. 8). And we find that:

![alt text](images/image-4.png)

- Without GSD: even with IID data, model heterogeneity alone induces significant oscillations (accuracy oscillations showed by black lines in Fig. 8a–d).
- With GSD: Oscillation magnitude is reduced significantly, enabling stable convergence (colored lines in Fig. 8a-d).

This demonstrates that even local models are trained with IID data, they still introduce client drift for their sparse trained parameters when aggregated. And GSD uniquely addresses instability intrinsic to submodel-based training, a challenge unaddressed by prior works like Federated Dropout[1] or FedRolex[2].

Thank you again for the critical feedback about the role GSD plays in our framework and we hope the revised analysis clarify its theoretical and empirical significance.

### Response to comment O2-1

>While the limitations in effectively covering a global model's parameter space are compelling, in scenarios with a sufficiently large number of clients, random sampling might still provide adequate coverage. More details on the problem setup--particularly how many clients are involved and how their data or capacity constraints manifest--would help contextualize the necessity of the proposed method.

We appreciate these critical points and we acknowledge the importance of clarifying the problem setup and assumptions underlying FedEcover.

#### 1. Necessity of FedEcover in Large-Client Scenarios

While it may seem intuitive that "sufficiently large" client populations could mitigate coverage gaps through random sampling, our empirical and theoretical analyses demonstrate that ***naive randomness is insufficient even at scale***. For example:

- Empirical evidence:
  - In our synthetic datasets with 100 clients (CIFAR-100/Tiny ImageNet), FedEcover also achieve better accuracy and more convergence speedup by 1–4× compared to FD-m (Table V).
  - In our 3237-client FEMNIST experiments (Table III, Table V), FedEcover achieves both accuracy (78.86%) and convergence speedup (2.82x) outperforming $\text{FD-m}^+$(78.12%, 2.12x) despite both methods leveraging large client populations, highlighting that **sheer client quantity cannot compensate for inefficient coverage**. This gap arises because FD-m’s memoryless sampling leaves specific parameters over-trained/under-trained due to uneven coverage frequency (Fig. 4c).

- Theoretical grounding:
  - In Section V-A, we prove that naive random sampling with replacement requires the expected number of rounds $\frac{\lambda}{d \cdot \overline{c}}(m + \mathop{\text{ln}}\frac{n}{1-q}) (1 < \lambda < 2)$ to cover all neurons at least $m$ times with a probability no less than $q$, whereas FedEcover's buffer-based sampling guarantees uniform coverage with $\frac{m}{d \cdot \overline{c}}$ rounds. This gap widens as layer size $n$ and required probability $q$ increases.

Thus, FedEcover remains necessary even in large-client regimes to ensure deterministic coverage efficiency.

#### 2. Problem Setup and Client Heterogeneity

We explicitly define our problem setup in Section VI-A, including:

- Client scale: evaluations span two regimes (Section VI-A-2)
  - small-client-scale: 10 clients (all participating)
  - large-client-scale: 100 clients (20% participation) for synthetic datasets and 3237 clients (10 clients/round) for FEMNIST
- Data heterogeneity (Section VI-A-3):
  - Synthetic non-IID partitioning via Dirichlet distribution $Dir_N(\alpha)$ for controlled experiments (Fig. 5).
  - Real-world non-IID data (FEMNIST) with natural heterogeneity.
- Capacity heterogeneity (Section VI-A-4):
  - Client capacities follow a skewed distribution (e.g., 50% low-capacity clients with $c=0.1$, client capacity profile is modeled with the added Definition 1 in Section III in the revised manuscript), reflecting real-world device diversity and the fact that low-capacity devices are usually more affordable thus account for ratios. ![alt text](images/image-5.png)

This setup ensures our method is tested under realistic and diverse conditions.

### Response to comment O2-2

>It would also be valuable to discuss any assumptions regarding client distribution or system constraints (e.g., computational resources, communication overhead) that might influence the decision to adopt this framework.

#### 1. Assumptions and Practical Constraints

FedEcover is designed with the following practical assumptions:

- **Resource heterogeneity**: clients have varying computational/memory/communication capacities, modeled by $c_i \in (0, 1]$ (Definition 1 in Section III).
- **No public data and no raw data information sharing**: Unlike knowledge distillation-based and representation-based methods, FedEcover requires no auxiliary datasets and does not expose information like representation-label pairs that might be sensitive.

#### 2. When to Adopt FedEcover?

FedEcover is most beneficial in scenarios where:

- Clients exhibit significant capacity disparities (e.g., mix of smartphones and IoT devices).
- The global model must preserve high expressiveness (no downscaling).
- Data is heterogeneous, requiring full parameter space utilization.

In homogeneous or small-model settings, simpler methods like FedAvg may suffice. However, our results show FedEcover’s superiority in realistic, large-scale heterogeneous environments.

We hope these clarifications underscore the necessity of our method and its relevance to practical FL deployments.

## Response to Reviewer #2

### Response to comment O1

>Currently, it seems to me that the proposed solution can only be used for CNN and DNN with possible skip connections between layers. It would be great if the authors could add some discussion on the scope of the proposed solution and the possibility of being applied to general scenarios.

Thank you for raising this important point regarding the applicability of our proposed method to broader model architectures. We recognize this as a valid concern and provide the following clarifications and discussions:

#### 1. Applicability to Transformer Models

Our submodel extraction scheme is directly applicable to **Feed-Forward Networks (FFNs)** in Transformers, as FFN consist of linear layers (i.e., fully connected layers) where neurons can be randomly sampled without replacement, similar to CNN/DNN archtectures. Since FFNs account for a significant portion of decoder block (commonly used in popular decoder-only language modeling models) parameters (e.g., ~66.63% in [GPT2-Small](https://jalammar.github.io/illustrated-gpt2/#part-3-beyond-language-modeling)), optimizing these layers via FedEcover would already yield substantial efficiency improvements.

For **attention** layers, while our current submodel extraction does not explicitly introduce the technical details of submodel extraction of attention head (e.g., query/key/value projections), the core idea of coverage-aware sampling could be extended by treating attention heads as modular units. For example, submodels could dynamically select subsets of attention heads while maintaining structural consistency across layers. We acknowledge this as a promising direction for future work.

#### 2. Scope, limitations and future work

While our current experiments focus on CNNs and ResNets (to align with common FL benchmarks), the principles of coverage-aware submodel extraction and step-size decay are architecture-agnostic. The primary constraints arise in architectures with tightly coupled parameters (e.g., weight-tied layers in Transformers), which **require specialized indexing strategies**.

We agree that exploring FedEcover’s adaptation to Transformers, and even more architectures is a valuable direction. While we defer empirical validation of these extensions to future studies, we will briefly mention their potential in the revised paper’s conclusion to emphasize the generality of our framework.

Thank you again for your insightful comment. We believe this discussion strengthens the paper’s relevance to broader federated learning scenarios while maintaining a focused contribution on CNN/DNN-based applications.

### Response to comment O2

>The fonts in the figures are too small. It would be better if these fonts are at least comparable to the font in text.

Thank you for your valuable feedback regarding the font sizes in the figures. We acknowledge that ensuring readability is crucial for effective communication of results. We have taken the following improvements:

1. All figures (including subplots, axis labels, legends, and annotations) have been enlarged to enhance visibility.
2. High-resolution vector formats (e.g., PDF/SVG) are used to prevent pixelation or distortion during scaling.

If specific figures still require adjustments, we are happy to refine them further. Thank you again for highlighting this issue. We believe these improvements significantly enhance the visual presentaion quality of the manuscript.

### Response to comment O3

>Currently, the authors only used simple benchmark datasets such as CIFAR and ImageNet and they also tried to simulate the data heterogeneity by partitioning these datasets. However, it would be better if they can perform experiments on real datasets in real scenarios.

Thank you for your constructive feedback on the experimental validation of our framework. We appreciate your emphasis on real-world applicability and provide the following clarifications and enhancements.

In addition to synthetic non-IID benchmarks (CIFAR and Tiny ImageNet), we have rigorously evaluated FedEcover on **FEMNIST** from the LEAF project[3], a **real-world federated dataset** with inherent heterogeneity. FEMNIST is constructed by partitioning the Extended MNIST dataset based on *real users/writers*, reflecting natural data skew across devices. Key characteristics include:

- User-specific data generation: each client corresponds to a unique writer, capturing real-world variations in handwriting styles.
- Skewed data distribution: the number of samples per client varies (e.g., some clients have <100 samples, others >400), mirroring practical FL scenarios. The detailed information can be accessed through [the LEAF site](https://leaf.cmu.edu/).
- Non-synthetic partitioning: no artificial Dirichlet sampling is applied, ensuring authenticity in client data distributions.

Our experiments (Table III, V and Fig. 7d) demonstrate that FedEcover achieves 78.86% accuracy and 2.82x performance improving speedup on FEMNIST, outperforming all baselines. This validates the framework’s effectiveness in realistic, user-driven FL environments. And we also utilize the FEMNIST dataset with its large client scale to validate the superiority of our submodel extraction in terms of balanced neuron coverage frequency in Fig. 4.

We fully agree that further validation on additional real-world datasets would strengthen the framework’s generalizability. In future work, we plan to extend FedEcover to more domains like federated NLP (using user-partitioned text corpora) and healthcare (e.g., MIMIC-III with patient-specific data silos).

Thank you again for your insightful suggestion. We believe our current experiments strike a balance between controlled analysis and real-world validation, while laying a foundation for broader applications.

## Response to Reviewer #4

### Response to comment O1

>The global step-size decay mechanism (GSD) relies on specific hyperparameters (e.g., decay coefficient, stop rounds), but the paper lacks a analysis of how sensitive the framework's performance is to these hyperparameters.

Thank you for highlighting the importance of analyzing the sensitivity of the Global Step-size Decay (GSD) mechanism to its hyperparameters. We fully agree that understanding the robustness of hyperparameter choices is critical for practical deployment. In response to your feedback, we have conducted a comprehensive sensitivity analysis (highlighted in Section VI-E and Fig. 12 in the revised manuscript).

![alt text](images/image-3.png)

We evaluated FedEcover with $\gamma \in \{ 0.8, 0.85, 0.9, 0.95 \}$ and $T_{ds} = \{ 100, 200 \}$ while keep $T_{di} = 10$ on CIFAR-100 (large-client-amount regime). Results in our experiments demonstrate:

- Robustness across different value combinations: FedEcover maintains stable performance, with no more than 2% accuracy variation across all combinations of hyperparameter values.
- Superiority across different value combinations: FedEcover consistently achieves the best accuracy improvements and creditable speedups performance across different hyperparameter values.

Our experiments confirm that FedEcover’s GSD mechanism is **not overly sensitive to hyperparameters**, and its default settings generalize well across scenarios.

We appreciate your suggestion to strengthen this analysis, which has significantly improved the work’s rigor.

### Response to comment O2

>The related work section provides a basic overview but does not sufficiently compare the proposed method with previous approaches in a detailed manner, which would strengthen the justification for the new framework.

Thank you for your valuable feedback on the Related Work section. We appreciate your suggestion to strengthen the comparative analysis, which is critical for highlighting the novelty and necessity of our framework. In the revised manuscript, we have strengthened the discussion by systematically categorizing prior methods and explicitly contrasting their limitations with FedEcover’s innovations.

The Related Work section is originally organized with two methodological branches (knowledge sharing-based methods and submode extraction-based methods) and we further classified existing approaches based on their technical characteristics (marked in red italicized text in the revised manuscript). We discuss each related work class's limitations in details (e.g., impractical assumptions, limited application scenarios) along with its technical characteristics, while summarize the difference (also the advantages or innovations) of our prposed method with red-highlighted text at the last of each subsection. Specifically, the summarizations of advantages of our proposed method compared with related work are:

- Compared with knowledge sharing-based methods, our approach explicitly addresses their limitations (details in the revised manuscript) by eliminating public/synthetic data dependencies through parameter aggregation mechanism. This also maintains efficient and privacy-preserving, for parameter containing more (while non-sensitive) information than representations (along with labels, which might be sensitive), as well as enabling bi-directional knowledge flow through collaborative client-server interactions.
- Compared with existing submodel extraction-based methods, our framework focus on general functionally-dependent submodel extraction technique and innovates by introducing coverage-aware, diversity-aware and structure-aware submodel sampling that maximizes parameter space utilization while incorporating step-size decaying aggregation to promote convergence speed as well as stability.

### Response to comment O3

>Some parts of the paper are dense and technical, making it difficult for non-expert readers to follow. The clarity of explanations, especially regarding technical innovations like submodel extraction and GSD, could be improved.

Thank you for your constructive feedback. We fully agree that technical innovations like submodel extraction and Global Aggregation Step-size Decay (GSD) require clearer explanations to improve accessibility. In the revised manuscript, we have made the following improvements to enhance readability and logical flow.

**1. Expression Refinement**: We streamline technical descriptions to eliminate redundancy and improve conciseness. Specifically:

- Submodel extraction
  - **Overview-first** introduction: In Section IV ("Proposed Framework"), we first provide an intuitive overview of submodel extraction (Fig. 1) before diving into technical details. This helps readers grasp the high-level workflow before encountering algorithmic specifics.
  - Following **first-why-then-how** patterns: when discuss the two main designs (coverage-aware randomization and structure-aware residual connection consistency) in sequence, we first present our design objectives or reasons, and then introduce technical details.
- Global Step-size Decay (GSD)
  - In Section IV-B, we first present how to aggregate parameters from heterogeneous submodels and analyze the instability issue intensified by model heterogeneity. Then we present our solution: by introducing a decaying step-size to control the aggregation aggressiveness. Then we explain how this step-size decay mechanism influence the convergence stability.

**2. Structural Reorganization**: we notice that the original structure of Section IV incorporates dense comparative analysis, which might increase cognitive burden when readers have not built up the overall understanding about our framework. Thus, we make reorganization on the paper structure to make it easier to follow. Specifically:

- **Dedicated comparative analysis**: the theoretical and empirical superiority analysis of our submodel extraction method over existing approaches (e.g., FedRolex, FD-m) is now consolidated in Section V, allowing readers to first grasp our core methodology before engaging in comparative discussions.

These revisions ensure that non-expert readers can follow the core ideas without getting lost in technicalities upfront, while experts can delve into details in later sections. We appreciate your feedback and hope the revised manuscript addresses your concerns effectively.

### Response to comment O4

>The font size in the figures is relatively small. The author should adjust the font size to be larger, improving visual presentation quality.

Thank you for your valuable feedback regarding the font sizes in the figures. We acknowledge that ensuring readability is crucial for effective communication of results. We have taken the following improvements:

1. All figures (including subplots, axis labels, legends, and annotations) have been enlarged to enhance visibility.
2. High-resolution vector formats (e.g., PDF/SVG) are used to prevent pixelation or distortion during scaling.

If specific figures still require adjustments, we are happy to refine them further. Thank you again for highlighting this issue. We believe these improvements significantly enhance the visual presentaion quality of the manuscript.


## References

[1] S. Caldas et al., "Expanding the Reach of Federated Learning by Reducing Client Resource Requirements", 2019.

[2] S. Alam et al., "FedRolex: Model-Heterogeneous Federated Learning with Rolling Sub-Model Extraction", 2022.

[3] S. Caldas et al., "LEAF: A Benchmark for Federated Settings", 2018.

[4] J. Mills et al., "Faster Federated Learning With Decaying Number of Local SGD Steps", 2023.

## Note to the Meta-Reviewer

We sincerely thank the reviewers and the meta-reviewer for their constructive feedback, which has significantly strengthened our manuscript. All comments labeled Oi have been thoroughly addressed in the revised paper, with detailed responses provided in this response letter.

Regarding the suggestion to include an additional realistic benchmark with the example [Federated-Benchmark](https://github.com/jiahuanluo/Federated-Benchmark.git), we fully acknowledge its value but regret to note that the specified dataset is no longer publicly accessible, and our repeated attempts to contact its administrators and authors via email have gone unanswered. Despite this limitation, we have expanded our empirical validation to include the FEMNIST dataset (3,237 clients with natural non-IID characteristics) and rigorous synthetic non-IID experiments, ensuring robust evaluation under real-world heterogeneity.

We deeply appreciate the reviewers’ time and insights, which have been invaluable in refining our work. Thank you for considering our revised submission.

Sincerely,

all authors
