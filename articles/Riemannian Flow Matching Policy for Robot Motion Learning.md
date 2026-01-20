The paper introduces the **Riemannian Flow Matching Policy (RFMP)**, a novel generative model designed for learning and synthesizing robot sensorimotor policies. While deep generative models, such as diffusion processes, have shown promise in robotic tasks, they often suffer from expensive inference due to the requirement of solving stochastic differential equations (SDEs)2. Furthermore, standard diffusion models face computational complexities when applied to Riemannian manifolds, which are essential for geometric awareness in robotic motion (e.g., orientation)3.


RFMP addresses these limitations by leveraging **Flow Matching (FM)**. By design, FM encodes high-dimensional multimodal distributions and offers a simpler, faster inference process compared to diffusion models4. The authors pioneer the application of FM to sensorimotor policy learning, demonstrating that RFMP provides smoother trajectories and lower inference times while respecting the intrinsic geometry of the robot's state space.

### **2. Theoretical Background**

#### **A. Riemannian Manifolds**

The work is grounded in Riemannian geometry, where a smooth manifold $\mathcal{M}$ (e.g., a globe surface) locally resembles Euclidean space $\mathbb{R}^d$. Key concepts utilized include:

- **Tangent Space ($\mathcal{T}_x\mathcal{M}$):** The vector space of tangent vectors at point $x$, isomorphic to $\mathbb{R}^d$.
    
- **Riemannian Metric ($g_x$):** A family of inner products allowing the definition of distances and angles on the manifold8.
    
- **Exponential and Logarithmic Maps:** The exponential map $Exp_x(u)$ maps a tangent vector to the manifold via geodesics, while the logarithmic map is its inverse9.
    

#### **B. Flow Matching (FM)**

FM is a simulation-free generative framework that transforms a simple base density $p_0$ to a target distribution $p_1$ via a flow $\phi_t$. The flow is defined by a vector field $u_t$ governed by the ordinary differential equation (ODE):

$$\frac{d\phi_t(x)}{dt} = u_t(\phi_t(x))$$

11. The model learns a parameterized vector field $v_t(x; \theta)$ by minimizing the regression loss against a target vector field, often utilizing a conditional flow matching (CFM) objective to make the problem tractable12121212.


#### **C. Riemannian Flow Matching (RCFM)**

For data on manifolds, the vector field evolves on the tangent bundle $\mathcal{TM}$13. The authors utilize the RCFM framework where the loss is computed using the Riemannian metric norm:

$$\mathcal{L}_{RCFM}(\theta) = \mathbb{E}_{t, q(z), p_t(x|z)} ||v_t(x; \theta) - u_t(x|z)||_{g_x}^2$$

14. The target vector field is typically designed using geodesic paths connecting samples $x_0$ and $x_1$15.


### **3. Methodology: The Riemannian Flow Matching Policy**

RFMP adapts the RCFM framework to learn a policy $\pi_\theta(a|o)$ from a set of demonstrations $\mathcal{D}=\{o_n, a_n\}$.

#### **Training Strategy**

- **Conditioning:** The vector field $v_t(a|o)$ is conditioned on an observation vector $o$.
    
- **Receding Horizon:** Inspired by diffusion policies, the action $a$ is an action horizon vector $[a_{\tau}, ..., a_{\tau+T_a}]$ for prediction horizon $T_a$, ensuring temporal consistency.
    
- **Observation Sampling:** The observation vector $o = [o_{\tau-1}, o_c, \tau-c]$ is constructed using a reference observation and a randomly sampled context observation $o_c$ to provide motion direction information.
    
    
- Loss Function: The specific RFMP loss minimizes the difference between the learned and target geodesic vector fields:
    $$\mathcal{L}_{RFMP}(\theta) = \mathbb{E}_{t, q(a_1), p_t(a|a_1)} ||v_t(a|o; \theta) - u_t(a|a_1)||_{g_a}^2$$
    

#### **Inference and Implementation**

- **Inference:** The policy is queried by sampling noise $a_0 \sim p_0$, solving the ODE using the learned vector field $v_t(a|o; \theta)$, and executing the first $T_e$ actions.
    
- **Architecture:** The vector field is parameterized by a simple Multilayer Perceptron (MLP) with 5 layers and 64 hidden units, totaling only 32K parameters.
    
- **Base Distributions:**
    
    - **Euclidean ($\mathbb{R}^2$):** Standard Gaussian $p_0 = \mathcal{N}(0, \sigma I)$.
        
    - **Riemannian ($\mathcal{S}^2$):** Wrapped Gaussian distribution centered at the manifold origin.
        

### **4. Experimental Evaluation**

The authors evaluated RFMP on the LASA dataset in Euclidean space ($\mathbb{R}^2$) and projected onto a sphere ($\mathcal{S}^2$), comparing it against Diffusion Policies (DP).

#### **A. Trajectory-Based Policies**

- **Performance:** RFMP closely reproduces demonstrations. When initialized with random observations, RFMP generalizes well, whereas DP tends to force trajectories back to the original training data support, exhibiting "memorization" behavior.
    
- **Smoothness:** RFMP generates significantly smoother trajectories (lower jerkiness) than DP. The authors hypothesize DP's jerkiness arises from the inherent stochasticity of diffusion inference.
    
- **Geometric Consistency:** On the sphere $\mathcal{S}^2$, DP fails to strictly adhere to the manifold geometry (trajectories entering the sphere), while RFMP naturally respects the manifold constraints.
    
- **Inference Time:** In Euclidean settings, RFMP is approximately 30% (~350ms) faster than DP because it uses ODE solvers rather than the more expensive SDE solvers. On the sphere, RFMP appears slower only because it employs specific Riemannian ODE solvers, whereas the baseline DP incorrectly uses Euclidean solvers, disregarding geometry.

#### **B. Visuomotor Policies**

- **Setup:** The policy was conditioned on latent encodings from raw grayscale images using a modified ResNet-18 backbone.
    
- **Results:** Visuomotor RFMP successfully reproduces demonstration patterns in both Euclidean and Riemannian settings.
    
- **Comparison:** RFMP performs competitively with DP regarding task accuracy (DTWD) but maintains a clear advantage in trajectory smoothness.
    
- **Speed:** Visuomotor RFMP showed a ~45% reduction in inference time compared to DP34.
    

### **5. Conclusion**

The paper concludes that Riemannian Flow Matching Policies (RFMP) offer a compelling alternative to diffusion models for robot motion learning. RFMP provides:

1. **Geometric Awareness:** Inherently handles data on manifolds35.
    
2. **Efficiency:** Significantly faster inference times using simpler network architectures (MLP vs. CNN).
    
3. **Quality:** Smoother action trajectories and robust performance even with short prediction horizons.