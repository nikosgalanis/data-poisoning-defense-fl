# Types of Federated Learning

Based on 

[1]: Zhang, C. et al. (2021) ‘A survey on federated learning’, Knowledge-Based Systems, 216, p. 106775. Available at: https://doi.org/10.1016/j.knosys.2021.106775.

[2]: Chen, Y. et al. (2022) ‘Federated Learning Attacks and Defenses: A Survey’. arXiv. Available at: https://doi.org/10.48550/arXiv.2211.14952.

## By Data Partition

### Horizontal FL
 - The essence of HFL is the **union of samples**. When users’ features of two datasets overlap more and users overlap less, we divide the dataset horizontally and take out the part of the data where the users’ features of both datasets are the same, but the users are not exactly the same for training (that is the user dimension). 

 - Example: Two hospitals in different locations. They have minimal intersection in their data samples, but their operations are quite alike. As a result, *the data samples exhibit many similarities*. In such a scenario, we can utilize HFL to construct a combined model. [2]
 
### Vertical FL
 - The essence of VFL is the **union of features**. When two datasets have more overlap in users but less overlap in users’ features, we divide the dataset vertically and take out the part of the data where the users on both sides are the same, but the user features are not exactly the same for training (that is the feature dimension)
  
 - Example: Bank and e-commerce company (same area). The user base of both institutions includes a significant number of local residents, leading to a *large overlap in their data samples*. The data collected by the bank are related to the credit level of the users, whereas the data collected by the e-commerce company relates to consumer behavior. As a result, the data characteristics of these two institutions are different. [2]


### Federated Transfer Learning

 - In those cases where both datasets have less overlap in users and users’ features, *we do not slice the data and use the transfer learning method to overcome the shortfall of data or label*. (**This is called Hybrid FL in [2]**)

 - We can use Transfer learning to overcome the lach of data or tags

 - Example: Bank located in China and e-commerce company located in the United States.[5] Geographical limitations result in limited overlap of the user groups of the two institutions. However, the limited feature space overlap due to distinct businesses. In such a scenario, *TL techniques can be utilized to generate solutions for the entire sample and feature space within a federation*.
  
 - **VERY INTERESTING** Generally, a handful of well-tuned and intricate centralized models (*Teacher*) pre-trained with large datasets are shared on public platforms, and individual users further customize accurate models (*Student*) for specific tasks using the pre-trained teacher model as a launching point via only limited training on the smaller domain-specific datasets [6].

 - Pre-trained teacher models could gradually become the more attractive and vulnerable target for attackers to manipulate, so that student models that use such maliciously manipulated teacher models can incur immense threats. **#IDEA: Observing the malicious data before inherited from teacher.**

##### Literature explaining the model
[3]: Liu, Y. et al. (2020) ‘A Secure Federated Transfer Learning Framework’, IEEE Intelligent Systems, 35(4), pp. 70–82. Available at: https://doi.org/10.1109/MIS.2020.2988525.

[4]: Pan, S.J. et al. (2010) ‘Cross-domain sentiment classification via spectral feature alignment’, in Proceedings of the 19th international conference on World wide web. WWW ’10: The 19th International World Wide Web Conference, Raleigh North Carolina USA: ACM, pp. 751–760. Available at: https://doi.org/10.1145/1772690.1772767.

[5]: Yang, Q. et al. (2019) ‘Federated Machine Learning: Concept and Applications’. arXiv. Available at: https://doi.org/10.48550/arXiv.1902.04885.

##### Literature providing attacks
 - In general, this categorization does not provide different types of attacks, as it is mainly about the data layer of the model, and not about its characteristics.
 - However, Transfer Learning has some attacks in literature, though *not in the federated setting.*
 - [6]: Wang, S. et al. (2022) ‘Backdoor Attacks Against Transfer Learning With Pre-Trained Deep Learning Models’, IEEE Transactions on Services Computing, 15(3), pp. 1526–1539. Available at: https://doi.org/10.1109/TSC.2020.3000900. (#TODO): Read in depth

## By methods for solving heterogeneity
 - Because of the variety of devices that take part in the FL process, the difference of equipment will eventually affect the training. To solve the problem of system heterogeneity, the following architectures are implemented.
 
 - Synchronous aggregation fails due to its waiting for straggler devices before aggregation in each training round.

### Asynchronous Federated Learning
 - In AFL, the global model aggregation happens whenever a new local model is received by the aggregation server.
 
 - **Node selection**: In AFL, it is more eager to select nodes that are more robust and powerful while preventing the global model from overfitting.

 - **Weighted Aggregation**: In AFL, the goal is to mitigate the effects of stale local models generated based on an outdated global model, which does not exist in classic FL. By incorporating a staleness parameter, weighted aggregation reduces the weight of stale local models and increases the weight of the most current local models during the aggregation procedure.

 - **Gradient Compression**: Since gradient compression is a general strategy to improve the efficiency of FL, it is usually adopted in AFL to further reduce communication costs. After introducing AFL, gradient compression faces new challenges of resource-constrained edge/IoT computing environment and more frequent aggregation operation.

##### Literature explaining the model
 - [7]: Xu, C. et al. (2022) ‘Asynchronous Federated Learning on Heterogeneous Devices: A Survey’. arXiv. Available at: https://doi.org/10.48550/arXiv.2109.04269.


### Semi-Asynchronous / Cluster Federated Learning
 - Clustered FL is an approach of increasing training efficiency by grouping together devices with similar performance, functionalities, or datasets. Inner-group update, inter-group update, or both could benefit from the asynchronous update strategy.

 - Framework proposal in [8]: CSAFL. CSAFL includes several groups, which can be deployed to the central server, or some devices in the middle layer, such as the edge server

##### Literature explaining the model
 - [8]: Zhang, Y. et al. (2021) ‘CSAFL: A Clustered Semi-Asynchronous Federated Learning Framework’. 2021 International Joint Conference on Neural Networks (IJCNN), pp. 1–10. Available at: https://doi.org/10.1109/IJCNN52387.2021.9533794.

### FL with Sampling
 - In some federated learning scenarios, the equipment **is selected** to participate in the training, while in another part of the scene, the equipment **takes the initiative** to participate in the training.
 - In [9] FedCS, mitigates the heterogenity problem while actively managing clients based on their resource conditions. Specifically, FedCS solves a client selection problem with resource constraints, which allows the server to aggregate as many client updates as possible and to accelerate performance improvement in ML models.
 - Using incentives: in [10], they adopt the contract theory to design an effective incentive mechanism for simulating the mobile devices with high-quality (e.g., high-accuracy) data to participate in federated learning.

#### Literature explaining the model
 - [9]: Nishio, T. and Yonetani, R. (2019) ‘Client Selection for Federated Learning with Heterogeneous Resources in Mobile Edge’, in ICC 2019 - 2019 IEEE International Conference on Communications (ICC), pp. 1–7. Available at: https://doi.org/10.1109/ICC.2019.8761315.
 - [10]: Kang, J. et al. (2019) ‘Incentive Design for Efficient Federated Learning in Mobile Networks: A Contract Theory Approach’. arXiv. Available at: https://doi.org/10.48550/arXiv.1905.07479.

#### Attacks
 - "Most privacy and robustness researches are focused on FL with homogeneous architectures. It remains unclear whether existing attacks, privacy-preserving techniques and defense mechanisms can be adapted to FL with heterogeneous architectures. It is valuable future work to explore similar types of attacks and defenses in heterogeneous FL." [11]
 - [11]: Lyu, L. et al. (2022) ‘Privacy and Robustness in Federated Learning: Attacks and Defenses’. arXiv. Available at: http://arxiv.org/abs/2012.06337 (Accessed: 20 January 2023).

### Peer to Peer FL

 - P2P FL is a distributed learning protocol without a central parameter server.
 - Each node has a set of peer nodes with whom the node can exchange model updates.
 - Currently there is no widely deployed P2PFL protocol or framework
 - Absence of a central server body not only makes our environment resistant to failure but also precludes the need for a body everyone trusts. [16]


#### Literature explaining the model
 - [16]: Roy, A.G. et al. (2019) ‘BrainTorrent: A Peer-to-Peer Environment for Decentralized Federated Learning’. arXiv. Available at: https://doi.org/10.48550/arXiv.1905.06731.
 - [17]: Lalitha, A. et al. (2019) ‘Peer-to-peer Federated Learning on Graphs’. arXiv. Available at: https://doi.org/10.48550/arXiv.1901.11173.


 #### Attacks
  - [18]: Yar, G., Nita-Rotaru, C. and Oprea, A. (2023) ‘Backdoor Attacks in Peer-to-Peer Federated Learning’. arXiv. Available at: https://doi.org/10.48550/arXiv.2301.09732.




## By data handling techniques

### FL with trusted hardware
 - Secure hardware, such as Intel-SGX and AMD enclave, provides a trusted environment for users to execute their code in an untrusted setting, even when assuming that the operating system itself may be compromised. [12]
 - non-judicious use of cryptographic primitives can lead to leakage of information
 - **Explanation**: At the beginning of the protocol, code is loaded into the enclaves and interested parties get a code attestation. Then, the active party’s enclave generates a key for a semantically secure symmetric encryption scheme (e.g., AES) and a secure channel is established between it and the passive parties’ enclaves. Secret keys are communicated (as necessary) via this secure channel.

#### Literature explaining the model
 - [12]: Chamani, J.G. and Papadopoulos, D. (2020) ‘Mitigating Leakage in Federated Learning with Trusted Hardware’. arXiv. Available at: https://doi.org/10.48550/arXiv.2011.04948.

### Agnostic FL
 - In [13] they propose a new framework of agnostic federated learning, where the centralized model is optimized for any target distribution formed by a mixture of the client distributions, instead of being biased towards different clients.
 - "Optimizes the centralized model for any target distribution formed by a mixture of the client distributions via a minimax optimization scheme". [14]

#### Literature explaining the model
 - [13]: Mohri, M., Sivek, G. and Suresh, A.T. (2019) ‘Agnostic Federated Learning’, in Proceedings of the 36th International Conference on Machine Learning. Available at: https://proceedings.mlr.press/v97/mohri19a.html
  
 - [14]: Li, T. et al. (2020) ‘Federated Learning: Challenges, Methods, and Future Directions’, IEEE Signal Processing Magazine, 37(3), pp. 50–60. Available at: https://doi.org/10.1109/MSP.2020.2975749.

## Special case: Split Learning

 - Split Learning is a distributed and private method for training deep neural networks developed by MIT Labs. It allows for training across multiple data sources without the need for sharing raw, labeled data directly.
 - In split learning, a deep neural network is **split into multiple sections**, each of which is trained on a different client.
 - The main disadvantage of FL is that each client needs to run the full ML model, and resource-constrained clients, such as available in the Internet of Things, could not afford to run the full model.
 - SL splits the full ML model into multiple smaller network portions and train them separately on a server, and distributed clients with their local data. Assigning only a part of the network to train at the client-side reduces processing load (compared to that of running a complete network as in FL), which is significant in ML computation on resource-constrained devices
 - The difference between SL and FL lies in the communication content between clients
and the server. In split learning, clients are responsible for training the shallower layers of the model, and update the outputs of shallower layers to the server.
 - [15] proposes a mix of FL and Split Learning
- 
 - limited literature after 2018

#### Literature explaining the model
 - didnt find the whitepaper or any other work from MIT explaining the details of the model
 - [MIT website for the paper](https://splitlearning.mit.edu/)
 - [15]: Thapa, C. et al. (2022) ‘SplitFed: When Federated Learning Meets Split Learning’. arXiv. Available at: http://arxiv.org/abs/2004.12088 (Accessed: 20 January 2023).