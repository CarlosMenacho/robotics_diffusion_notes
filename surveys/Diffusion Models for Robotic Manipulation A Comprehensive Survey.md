This survey provides the first comprehensive review of diffusion models (DMs) applied to robotic manipulation, covering state-of-the-art methods across trajectory planning, grasp synthesis, and visual data augmentation. The authors systematically categorize methodologies based on network architecture, learning frameworks, applications, and evaluation metrics.

---

## 1. Introduction

### 1.1 Motivation

**Key Advantages of Diffusion Models:**

- **Multi-modal distribution modeling**: Superior ability to capture multiple valid solutions (e.g., multiple feasible trajectories or grasp poses)
- **High-dimensional robustness**: Effective handling of complex visual inputs (images, point clouds) and action spaces
- **Training stability**: More stable than GANs, avoiding mode collapse issues
- **Performance**: Surpass Gaussian Mixture Models (GMMs) and Energy-Based Models (EBMs) in practice

### 1.2 Historical Context

- **Pre-2022**: Limited use of probabilistic models in robotics
- **2022-present**: Rapid adoption following breakthroughs in computer vision (DALL-E, Stable Diffusion)
- **Trend**: Exponential growth in robotics applications, particularly in manipulation tasks

---

## 2. Mathematical Foundations

### 2.1 Core Framework

#### 2.1.1 Denoising Score Matching with Langevin Dynamics (SMLD)

**Forward Process:**

- Data x from distribution p_data(x) is gradually corrupted by adding noise
- Noise schedule {σ_k} with progressively increasing variance
- Perturbed distribution: p_σk(x_k | x)

**Training Objective:**

$$

\mathcal{L} = \frac{1}{2K} \sum_{k} \sigma_k^2 \,
\mathbb{E}\!\left[
\left\|
\nabla_{x_k} p_{\sigma_k}(x_k \mid x)
- s_\theta(x_k, \sigma_k)
\right\|^2
\right]

$$
**Reverse Process:**

- Use Langevin dynamics with learned score function
- Iteratively denoise from random noise to clean samples

#### 2.1.2 Denoising Diffusion Probabilistic Models (DDPM)

**Forward Process (Closed Form):**

$$
p(x_k \mid x_0) = \mathcal{N}\!\left(
x_k;\,
\sqrt{\bar{\alpha}_k}\, x_0,\,
(1 - \bar{\alpha}_k)\mathbf{I}
\right)
$$

**Training Objective:**


$$
\mathcal{L} = \mathbb{E}\!\left[
\left\|
\varepsilon - \varepsilon_\theta(x_k, k)
\right\|^2
\right]
$$

**Reverse Process:**

$$
x_{k-1}
= \frac{1}{\sqrt{\alpha_k}}
\left(
x_k
- \frac{1 - \alpha_k}{\sqrt{1 - \bar{\alpha}_k}}
\, \varepsilon_\theta(x_k, k)
\right)
+ \sigma_k z
$$
### 2.2 Architectural Improvements

#### Key Methods for Faster Sampling:

1. **DDIM (Denoising Diffusion Implicit Models)**
    - Deterministic sampling process
    - Reduces steps from 1000 (training) to 10-100 (inference)
    - Most commonly used in robotics applications
2. **DPM-Solver**
    - Second-order ODE solver
    - Non-uniform step sizes for better efficiency
    - Superior performance with few sampling steps
3. **Flow Matching**
    - Learns velocity field directly instead of noise
    - More stable training, fewer hyperparameters
    - Risk of mode collapse with few steps (mitigated by self-consistency)

### 2.3 Adaptations for Robotic Manipulation

#### 2.3.1 Conditioning on Observations

- **Visual observations**: RGB images, RGB-D, point clouds
- **State information**: Joint angles, end-effector poses
- **Language instructions**: Natural language task descriptions
- **Multi-modal fusion**: Combining visual, tactile, and linguistic inputs

#### 2.3.2 Receding Horizon Control

- Generate trajectory subsequences instead of complete trajectories
- Execute partial plan, then replan based on updated observations
- Balances planning horizon with reactivity to environment changes

**Key Parameters:**

- **Planning horizon (H)**: Length of predicted trajectory
- **Control horizon (H_c ≤ H)**: Number of steps executed before replanning
- Prevents error accumulation while maintaining temporal coherence

---

## 3. Network Architectures

### 3.1 Convolutional Neural Networks (CNNs)

#### 3.1.1 Temporal U-Net

- **Origin**: Adapted from image generation to trajectory planning by Janner et al. (2022)
- **Modification**: Replace 2D spatial convolutions with 1D temporal convolutions
- **Applications**: Trajectory generation, action sequence modeling

#### 3.1.2 Diffusion Policy Architecture (Chi et al., 2023)

- **Key innovation**: FiLM (Feature-wise Linear Modulation) conditioning
- **Conditioning method**:
    - History of observations encoded via CNN
    - Features modulated at each U-Net layer
    - Enables multi-modal and constraint integration
- **Advantages**: Lower hyperparameter sensitivity, effective for positional control

### 3.2 Transformers

#### 3.2.1 Architectures

- **Multi-head cross-attention**: Most common variant
- **Diffusion Transformers (DiT)**: Specialized transformer-based denoising networks
- **Input tokens**: Observation history, denoising timestep, noisy actions
- **Conditioning**: Via self-attention and cross-attention mechanisms

#### 3.2.2 Advantages

- Superior long-range dependency modeling
- Better handling of high-dimensional data
- Excellent for long-horizon tasks requiring high-level reasoning

#### 3.2.3 Trade-offs

- Higher computational cost than CNNs
- Longer inference times
- More sensitive to hyperparameters than U-Nets

### 3.3 Multi-Layer Perceptrons (MLPs)

#### 3.3.1 Characteristics

- **Architecture**: 2-4 hidden layers with Mish activation
- **Primary use**: Reinforcement learning applications
- **Advantages**: Computational efficiency, training stability
- **Limitations**: Poor performance on high-dimensional visual inputs

#### 3.3.2 Applications

- Offline RL with ground-truth state information
- Q-learning integration (Diffusion-QL)
- Skill-based learning

### 3.4 Architecture Comparison

|Architecture|Computational Efficiency|Visual Processing|Long-Horizon Tasks|Hyperparameter Sensitivity|
|---|---|---|---|---|
|CNN (U-Net)|Medium|Good|Medium|Low|
|Transformer|Low|Excellent|Excellent|High|
|MLP|High|Poor|Poor|Low|

---

## 4. Applications

### 4.1 Trajectory Generation

#### 4.1.1 Imitation Learning

##### 4.1.1.1 Action and Pose Representations

**Task Space (Most Common):**

- End-effector positions (translation + rotation)
- Representations: Euler angles, quaternions
- Micro-actions with close proximity enable simple positional controllers

**Joint Space:**

- Direct joint angle prediction
- Advantages: Avoids singularities, direct motor control
- Less common due to higher dimensionality

**Image Space:**

- Predict future image sequences showing robot/object motion
- Advantages: Leverage large-scale video data, embodiment-agnostic
- Challenges: Temporal consistency, computational cost, physical plausibility

**Examples:**

- Ko et al. (2024): Video sequence generation
- Bharadhwaj et al. (2024): Point tracking in image space

##### 4.1.1.2 Visual Data Modalities

**2D Visual Representations:**

- RGB images (single or multi-view)
- Common encoders: ResNet, Vision Transformers (ViT)
- Advantages: Simpler setup, lower computational cost

**3D Visual Representations:**

- Point clouds (single or multi-view)
- RGB-D fusion
- Encoders: PointNet++, VN-DGCNN
- **Performance gains**:
    - 3D Diffusion Policy (Ze et al., 2024): +24.2% over 2D on complex tasks
    - Better geometric understanding for manipulation

##### 4.1.1.3 Long-Horizon and Multi-Task Learning

**Hierarchical Approaches:**

- **Skill-based**: Multiple skill-specific DMs + high-level planner
    - Examples: Mishra et al. (2023), Liang et al. (2024)
    - High-level planner: VAE, regression model, or skill selector
- **Goal-conditioned**: Coarse-to-fine planning
    - High-level policy predicts subgoals
    - Low-level DM generates trajectories to subgoals
    - Examples: Ma et al. (2024), Du et al. (2023)

**Multi-Modal Prompting:**

- Language instructions (natural language task descriptions)
- Video demonstrations
- Tactile feedback
- Combined modalities for versatile skill chaining

##### 4.1.1.4 Vision-Language-Action Models (VLAs)

**Motivation:**

- Pretrained on internet-scale data
- Strong generalization capabilities
- Challenges: Slow inference, imprecise actions

**Integration Strategies:**

1. **Diffusion Refinement:**
    - VLA generates coarse actions
    - DM refines for precision and embodiment adaptation
    - Example: TinyVLA (Wen et al., 2025)
2. **Flow Matching Alternative:**
    - More efficient than diffusion for VLA integration
    - Single-step inference possible
    - Examples: π0 (Black et al., 2024), Zhang & Gienger (2025)
3. **Architecture:**
    - VLA backbone encodes observations + language
    - Diffusion decoder generates actions
    - Reasoning injection modules for logic-to-action translation

##### 4.1.1.5 Constrained Planning

**Classifier Guidance:**

- Train separate model to score constraint satisfaction
- Add gradient of score to denoising process
- Challenges: Requires additional training, computational overhead

**Classifier-Free Guidance:**

- Train conditional and unconditional DMs in parallel
- Sample from weighted mixture during inference
- Advantages: No gradient computation, arbitrary constraint combinations
- Limitations: Doesn't generalize to unseen constraints

**Constraint Satisfaction Methods:**

- **Inpainting**: Replace trajectory states with constraint values
    - Simple but limited to point-wise constraints
    - Can push trajectories into low-likelihood regions
- **Constraint Tightening** (Römer et al., 2024):
    - Integrate constraints directly into reverse process
    - Guarantees satisfaction vs. guidance methods
    - Limited real-world evaluation

#### 4.1.2 Reinforcement Learning

##### 4.1.2.1 Offline RL Integration

**Diffuser (Janner et al., 2022) - Classifier-Based:**

- Train return prediction model R_φ(τ_k)
- Add guidance term to sampling:


$$
p(\tau_{k-1} \mid \tau_k, O)
\approx
\mathcal{N}\!\left(
\tau_{k-1};
\mu + \Sigma \nabla R_\phi(\mu),
\Sigma
\right)
$$
- Limitations: Policy trained independently of reward signal

**Decision Diffuser (Ajay et al., 2023):**

- Condition DM directly on trajectory return
- Uses classifier-free guidance
- Outperforms Diffuser on block-stacking tasks
- Limitation: Similar to imitation learning, limited generalization

**Diffusion-QL (Wang et al., 2023):**

- Integrate Q-learning with diffusion
- Train Q-function with Bellman operator
- Add policy improvement to diffusion loss:


$$
\mathcal{L}_{\mathrm{RL}}
=
\mathcal{L}_{\mathrm{diffusion}}
+
\alpha \, \mathbb{E}\!\left[
- Q_\phi(s, a_0)
\right]
$$


- Better alignment with optimal behavior

##### 4.1.2.2 Skill Composition in RL

- Multiple skill-specific DMs for low-level control
- High-level policy trained with online RL
- Enables long-horizon task learning from suboptimal data

##### 4.1.2.3 Limitations

- Mostly offline methods (distribution shift vulnerability)
- Limited real-world evaluation
- Few methods process visual observations
- Online RL with DMs largely unexplored

### 4.2 Grasp Generation

#### 4.2.1 Diffusion on SE(3) Grasp Poses

**Challenge:** Standard diffusion operates in Euclidean space, but grasp poses live on SE(3) manifold

**SE(3)-Diff (Urain et al., 2023):**

- Score matching on Lie groups
- Energy-based model for grasp quality
- Advantages: Direct quality evaluation, integration with motion planning
- Limitations: Extensive sampling requirements, generalization challenges

**Flow Matching Approaches:**

- **EquiGraspFlow** (Lim et al., 2024): CNF-based, preserves SE(3) equivariance
- **Grasp Diffusion Network** (Carvalho et al., 2024): No auxiliary supervision needed
- Advantages: More efficient training than SE(3)-Diff

**Bi-Equivariance:**

- Ryu et al. (2024): Equivariant descriptor fields extended to diffusion
- Transformations in both observation and end-effector frames
- Improves sample efficiency
- Freiberg et al. (2025): Extended to multi-embodiment grasping

#### 4.2.2 Latent Space Diffusion

**GraspLDM (Barad et al., 2024):**

- VAE-based latent diffusion
- Conditioned on point cloud and task latent
- Trade-off: Implicit modeling may reduce geometric consistency

#### 4.2.3 Applications

**Parallel-Jaw Grasping:**

- Single grasp pose prediction
- 6-DoF (position + orientation)
- Common benchmarks: Acronym, VGN, DA2

**Dexterous Grasping:**

- Multi-fingered hands
- High-dimensional joint spaces
- Examples: DexGraspNet, MultiDex benchmarks
- Approaches: Wu et al. (2024), Weng et al. (2024)

**Language-Guided Grasping:**

- Natural language specifies grasp intent or object parts
- Integration with CLIP, foundation models
- Examples: Nguyen et al. (2024), Vuong et al. (2024)

**Affordance-Based Manipulation:**

- Object pose diffusion for rearrangement
- Pre-grasp manipulation via imitation learning
- Task-oriented reorientation

### 4.3 Visual Data Augmentation

#### 4.3.1 Dataset Scaling

**Methods:**

- **Object augmentation**: Change colors, textures, replace objects
- **Background augmentation**: Increase robustness to distractors
- **Trajectory augmentation**: Modify object positions + adapt actions (rare)
- **Language augmentation**: Generate corresponding task descriptions

**Approaches:**

1. **Frozen pretrained models**: Stable Diffusion, DALL-E
2. **Fine-tuned models**: Domain-specific adaptations
3. **Inpainting**: Semantically meaningful scene modifications

**Advantages over Domain Randomization:**

- Grounded in real-world data
- Less manual tuning per task
- Semantically coherent augmentations

**Examples:**

- Chen et al. (2023): GenAug for behavior retargeting
- Zhang et al. (2024): DAgger with off-distribution demos
- Di Palo et al. (2024): Hindsight experience replay with visual alignment

#### 4.3.2 Scene Reconstruction

**Viewpoint Reconstruction:**

- Generate novel views from single RGB-D image
- Methods: Kasahara et al. (2024) using DALL-E + depth prediction
- Applications: Complete point clouds, handle occlusions

**View Planning:**

- Generate minimal set of informative viewpoints
- Pan et al. (2024): DM generates geometric priors → NeRF reconstruction
- Reduces movement cost while ensuring coverage

**Limitations:**

- High computational cost
- Cannot handle completely occluded objects
- Limited adoption in manipulation (more common in computer vision)

#### 4.3.3 Object Rearrangement

**Task:** Generate target object arrangements from language prompts

**Evolution:**

1. **Early methods**: Zero-shot with DALL-E
    - Limitations: Scene inconsistencies, poor geometric understanding
2. **Modern approaches**:
    - Combine LLMs + VLMs (CLIP) + vision methods (NeRF, SAM)
    - Custom-trained DMs for scene-specific arrangements
    - Examples: Xu et al. (2024), Kapelyukh et al. (2024)

**Related:** Object pose diffusion (Section 4.2)

- Focus here: Multi-object rearrangement from sparse language
- All methods demonstrate real-robot effectiveness

---

## 5. Experimental Evaluation

### 5.1 Common Benchmarks

#### Imitation Learning:

- **CALVIN**: Language-conditioned long-horizon tasks
- **RLBench**: Multi-task manipulation benchmark
- **Relay Kitchen**: Sequential manipulation skills
- **Meta-World**: Diverse manipulation primitives
- **LIBERO**: Lifelong learning
- **FurnitureBench**: Real-world furniture assembly

#### Reinforcement Learning:

- **D4RL Kitchen**: Offline RL manipulation
- **Adroit**: Dexterous manipulation
- **KUKA**: Custom manipulation tasks

#### Grasp Learning:

- **Acronym**: Parallel-jaw grasping
- **VGN**: Volumetric grasp detection
- **DexGraspNet**: Dexterous grasping
- **MultiDex**: Multi-fingered manipulation

### 5.2 Common Baselines

**Diffusion-Based:**

- **Diffusion Policy (DP)** - Chi et al., 2023: Most common baseline for IL
- **3D Diffusion Policy** - Ze et al., 2024: Baseline for 3D representations
- **SE(3)-Diffusion** - Urain et al., 2023: Grasp generation baseline
- **Diffuser** - Janner et al., 2022: RL baseline
- **Diffusion-QL** - Wang et al., 2023: Offline RL baseline
- **Decision Diffuser** - Ajay et al., 2023: Outperforms Diffuser on manipulation

**Non-Diffusion:**

- Behavior Cloning (BC)
- Implicit Behavior Cloning (IBC)
- GMMs
- GANs
- VAEs

### 5.3 Key Performance Results

**3D Diffusion Policy (Ze et al., 2024):**

- **Simulation**: 74.4% avg success rate (+24.2% over 2D DP)
- **Real-world**: 85.0% success rate (+50% over DP)
- **Tasks**: Dexterous manipulation (rolling dumpling, drilling, pouring)

**3D Diffuser Actor (Ke et al., 2024):**

- Significantly outperforms 3D-DP on CALVIN
- Especially strong on zero-shot long-horizon tasks
- Uses language + multi-view point clouds

**Decision Diffuser (Ajay et al., 2023):**

- Outperforms Diffuser on block-stacking, rearrangement
- Only evaluated in simulation

### 5.4 Evaluation Characteristics

**Simulation vs. Real-World:**

- Majority: Both simulation and real-world
- Some RL methods: Simulation only
- Zero-shot sim-to-real: Increasing (using domain randomization, scene reconstruction)

**Data Requirements:**

|Method Category|Typical #Demos|Notes|
|---|---|---|
|Modern IL|5-50|With 3D/foundation models|
|Traditional IL|500-5000|2D visual observations|
|Offline RL|10k-6.5M transitions|Suboptimal data acceptable|
|VLA-based|400k videos|Internet-scale pretraining|

**Training Resources:**

- VLA methods: 100+ GPUs, massive datasets
- Standard IL: Single GPU, hours to days
- RL methods: Higher computational cost than IL

---

## 6. Limitations and Future Directions

### 6.1 Current Limitations

#### 6.1.1 Generalizability

**Challenges:**

- **Covariate shift**: IL methods struggle with out-of-distribution states
- **Distribution shift**: Offline RL cannot adapt to new scenarios
- **Limited transfer**: Most methods require retraining for new tasks

**Partial Solutions:**

- Foundation models (VLAs) for broad generalization
- Data augmentation (but limited to observation space)
- Skill composition (but requires predefined skill sets)

#### 6.1.2 Sampling Speed

**Current State:**

- DDIM most common: 10-100 steps
- DPM-Solver shows promise but limited evaluation in robotics
- Real-time control achievable (0.1s on Nvidia 3080) but still slower than direct methods

**Recent Advances:**

- Flow matching: Potentially single-step inference
- Consistency models: Few-step distillation
- BRIDGeR (Chen et al., 2024): Informed initialization

**Impact:**

- Critical for VLA integration (both are slow)
- Trade-off between speed and sample quality
- Task-dependent requirements (manipulation vs. navigation)

#### 6.1.3 Incomplete Evaluations

- Few comparisons of different samplers (DDIM vs DPM-Solver vs others)
- Limited ablations on number of denoising steps
- Sparse benchmarking across methods
- Few standardized comparison protocols

### 6.2 Promising Research Directions

#### 6.2.1 Continual and Lifelong Learning

**Current State:**

- Largely unexplored for DMs in robotics
- Existing work (Di Palo et al., 2024; Mendez-Mendez et al., 2023) has severe limitations

**Challenges:**

- Catastrophic forgetting prevention
- Efficient memory management (avoid replaying all past data)
- Skill accumulation without predefined abstractions
- Dynamic environment adaptation

**Potential Approaches:**

- Regularization techniques from continual learning literature
- Memory consolidation strategies
- Modular architectures for skill retention

#### 6.2.2 Improved Constraint Satisfaction

**Needs:**

- Guaranteed constraint satisfaction (beyond soft guidance)
- Generalization to unseen constraints
- Multi-constraint composition
- Real-world validation beyond simulation

**Promising Work:**

- Römer et al. (2024): Constraint tightening (needs broader evaluation)
- Movement primitive integration (Scheikl et al., 2024)

#### 6.2.3 Enhanced Scene Understanding

**For Complex/Cluttered Environments:**

- View planning combined with 3D DMs
- Iterative perception-planning loops
- Complete occlusion handling
- VLA integration for semantic reasoning

#### 6.2.4 Sampling Efficiency

**Priorities:**

- Systematic evaluation of modern samplers (DPM-Solver, flow matching) on manipulation tasks
- Domain-specific optimizations (e.g., BRIDGeR's informed initialization)
- Architecture improvements for faster inference
- Distillation techniques (teacher-student)

#### 6.2.5 Online Reinforcement Learning

**Current Gap:**

- Almost all DM+RL work is offline
- Limited online RL exploration
- Few methods for offline-to-online transition

**Opportunities:**

- Leverage DMs' multi-modal distribution modeling for exploration
- Combine with skill learning for sample efficiency
- Address sim-to-real gap with online fine-tuning

#### 6.2.6 Standardization

**Needed:**

- Unified benchmarking protocols
- Standardized baseline comparisons
- Reproducibility guidelines
- Open-source implementations

---

## 7. Key Insights and Recommendations

### 7.1 When to Use Diffusion Models

**Strong Fit:**

- Multi-modal action spaces (multiple valid solutions)
- High-dimensional visual observations
- Long-horizon tasks requiring temporal coherence
- Tasks with complex constraint combinations
- When training data exhibits diverse behaviors

**Poor Fit:**

- Time-critical applications without optimization
- Simple, single-mode distributions
- Limited computational budget
- When direct policy methods suffice

### 7.2 Architecture Selection Guidelines

**Choose CNNs (U-Net) when:**

- Need low hyperparameter sensitivity
- Positional control is primary mode
- Moderate computational budget
- 1D temporal structure dominates

**Choose Transformers when:**

- High-dimensional multi-modal inputs
- Long-horizon reasoning required
- Velocity control needed
- Sufficient computational resources

**Choose MLPs when:**

- RL with low-dimensional state
- Maximum computational efficiency required
- No visual processing needed
- Training stability is critical

### 7.3 Learning Framework Selection

**Imitation Learning:**

- High-quality demonstrations available
- Known successful behaviors
- Faster training, lower computational cost
- Accept covariate shift limitations

**Offline RL:**

- Suboptimal or diverse behavior data
- Need to model full state-action space
- Can afford careful hyperparameter tuning
- Accept higher computational cost

**Online RL / VLAs:**

- Need broad generalization
- Can leverage internet-scale pretraining
- Willing to trade precision for versatility
- Can combine with specialized refinement

### 7.4 Practical Implementation Tips

1. **Start Simple:**
    - Begin with DDIM (10-100 steps)
    - Use established architectures (DP, 3D-DP)
    - Validate on simulation before real-world
2. **Visual Observations:**
    - 3D representations when geometry is critical
    - Multi-view when single-view insufficient
    - Consider foundation model features (CLIP, DINOv2)
3. **Constraints:**
    - Classifier-free guidance for known constraints
    - Start with simple inpainting when applicable
    - Validate satisfaction rates carefully
4. **Sampling:**
    - Profile inference time for your application
    - Ablate denoising steps systematically
    - Consider flow matching for VLA integration

---

## 8. Conclusion

Diffusion models have emerged as a transformative technology for robotic manipulation, offering unparalleled capabilities in:

- Multi-modal distribution modeling
- High-dimensional data processing
- Stable training dynamics
- Versatile conditioning mechanisms

**Current State (2025):**

- Mature applications in trajectory planning and grasp synthesis
- Growing integration with foundation models (VLAs)
- Real-world deployments demonstrating practical viability
- Active research in sampling efficiency and generalization

**Critical Gaps:**

- Continual learning largely unexplored
- Online RL integration minimal
- Standardized evaluation limited
- Constraint satisfaction needs refinement

**Future Outlook:** The field is positioned for significant advances through:

- Hybrid approaches (DMs + VLAs + specialized modules)
- Improved sampling efficiency (flow matching, distillation)
- Enhanced scene understanding (3D representations + semantic reasoning)
- Lifelong learning capabilities

**Recommendation for Practitioners:** Diffusion models are production-ready for multi-modal manipulation tasks with moderate timing constraints. For critical applications requiring guaranteed constraint satisfaction or real-time control, hybrid approaches combining DMs with classical planning or faster alternatives (flow matching) are advisable.