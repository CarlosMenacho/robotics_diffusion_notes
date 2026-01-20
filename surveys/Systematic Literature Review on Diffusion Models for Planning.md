This paper presents a Systematic Literature Review (SLR) regarding the application of **Diffusion Models**—originally dominant in image generation—to the field of **Planning**. The review analyzes 41 relevant papers published between 2020 and May 2024, identifying a significant surge in related publications starting in 2023. The authors categorize the literature into fundamental planning, skill/conditional planning, safety/robustness, and domain-specific applications.

---

### **1. Datasets and Benchmarks**

The review identifies several key datasets used to evaluate diffusion-based planners, categorized by task type:

- **Motion and Path Planning:**
    
    - **D4RL:** A suite for offline reinforcement learning (RL) containing tasks like Maze2D and locomotion (e.g., HalfCheetah).
        
    - **Gym-MuJoCo:** Simulated continuous control environments for testing dynamic movement.
        
    - **AntMaze:** Specifically challenges path planning and navigation in constrained mazes.
        
    - **Robotic Manipulation:** Includes **Franka Kitchen** and **Adroit** for complex, multi-step tasks (e.g., opening microwaves) and dexterous manipulation.
        
- **3D and Autonomous Driving:**
    
    - **PROX & ScanNet:** Used for 3D scene understanding and grasp planning.
        
    - **nuPlan:** A benchmark for closed-loop planning in autonomous driving, assessing real-time decision-making.
        
- **Instructional Video:** Datasets like **CrossTask** and **COIN** are used to generate action sequences from visual observations.
    

---

### **2. Fundamental and General Planning**

This section covers foundational models and improvements in efficiency and replanning.

- **Foundational Models:**
    
    - **Diffuser:** A pioneering model-based RL approach that integrates trajectory optimization directly into the learning process, predicting all timesteps concurrently via iterative denoising.
        
    - **Diffusion-QL:** Focuses on offline RL by using a conditional diffusion model for policy representation. It combines behavior cloning with Q-learning guidance to generate high-value actions.
        
        
- **Motion and Path Planning:**
    
    - **EDMP:** Uses an ensemble of cost functions to guide diffusion, enhancing trajectory diversity in robotic manipulation.
        
        
    - **Motion Planning Diffusion:** Accelerates planning by sampling from a posterior distribution using a temporal U-Net.
        
        
    - **DiPPeR:** An image-conditioned planner for legged robots (e.g., Boston Dynamics’ Spot) that prioritizes speed and scalability.
        
        
- **Efficiency:**
    
    - **Latent Diffuser:** Improving efficiency by planning in a continuous latent action space rather than high-dimensional raw space.
        
    - **DiffuserLite:** Achieves real-time planning by generating coarse-to-fine trajectories to minimize redundancy.
        
- **Replanning:**
    
    - **RDM (Replanning with Diffusion Models):** Introduces a principled method to calculate the likelihood of a plan's executability, dynamically deciding when to replan in stochastic environments.
        

---

### **3. Skill Learning and Conditional Planning**

This category focuses on high-level skill abstraction and conditioning plans on specific contexts.

- **Skill Discovery:**
    
    - **XSkill:** A cross-embodiment framework that extracts skill prototypes from human videos to generate robot actions, bridging the gap between human demonstrations and robot execution.
        
    - **Generative Skill Chaining (GSC):** Uses a probabilistic framework to capture skill preconditions and effects, utilizing forward and backward diffusion for long-horizon tasks.
        
    - **SkillDiffuser:** Uses interpretable skills to condition a diffusion model, aligning state trajectories with high-level abstractions from language or vision.
        
        
- **Conditional Planning:**
    
    - **MetaDiffuser:** Designed for offline meta-RL, it uses a context-conditioned model to generalize to unseen tasks without environment interaction.
        
    - **Decision Diffuser:** Simplifies policy generation by using a return-conditional model, eliminating the need for dynamic programming/value function estimation.
        

---

### **4. Safety and Robustness**

Research in this area ensures plans are feasible, safe, and robust against uncertainty.

- **Safety Mechanisms:**
    
    - **SafeDiffuser:** Integrates **Control Barrier Functions (CBFs)** into the diffusion process to ensure safety constraints are consistently met.
        
    - **Cold Diffusion on the Replay Buffer (CDRB):** Optimizes planning using a replay buffer of "known good states" to improve obstacle avoidance.
        
    - **LTLDOG:** Ensures adherence to symbolic and temporal constraints (Linear Temporal Logic) during trajectory generation.
        
- **Managing Uncertainty:**
    
    - **Planning as In-Painting:** Treats planning in partially observable environments as an in-painting task, jointly modeling state trajectory and goal estimation.
        
    - **PlanCP:** Combines **conformal prediction** with diffusion dynamics to provide uncertainty estimates for robust trajectory prediction.
        

---

### **5. Domain-Specific Planning**

Diffusion models have been tailored for specific complex applications:

- **3D Planning:** **Scene Diffuser** combines scene-conditioned generation with physics-based optimization. **3D Diffusion Policy (DP3)** uses sparse point clouds for visual imitation learning in robotic manipulation.
    
    
- **Instructional Videos:** **PDPP** generates action sequences from start to goal states without intermediate supervision. **ActionDiffusion** incorporates temporal dependencies to improve action plan precision.
    
    
- **Autonomous Driving:** **Diffusion-ES** combines evolutionary search with diffusion to handle non-differentiable objectives and zero-shot instruction following, outperforming traditional planners on the nuPlan benchmark.
    
- **Multi-Tasking:** **MTDIFF** is designed to model and synthesize multi-task trajectory data for offline RL.
    

---

### **6. Research Challenges and Future Directions**

The review concludes by identifying critical hurdles for future research:

- **Hybrid Models:** Combining diffusion with **VAEs** could mitigate overfitting, while combining with **GANs** could accelerate inference speeds.
    
    
- **Scalability & Real-Time Use:** There is a need for pruning, quantization, and knowledge distillation to run these models on edge devices with low latency.
    
    
- **Generalization:** Models must avoid overfitting to training tasks; cross-domain validation and meta-learning strategies are suggested to improve robustness.
    
    
- **Human-Robot Interaction (HRI):** Future models should enhance interpretability and the ability to process ambiguous natural language instructions to improve collaboration.