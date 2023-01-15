# Types of Federated Learning

Based on 

[1]: Zhang, C. et al. (2021) ‘A survey on federated learning’, Knowledge-Based Systems, 216, p. 106775. Available at: https://doi.org/10.1016/j.knosys.2021.106775.

[2]: Chen, Y. et al. (2022) ‘Federated Learning Attacks and Defenses: A Survey’. arXiv. Available at: https://doi.org/10.48550/arXiv.2211.14952.

## Data Partition

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

#### Literature explaining the model
[3]: Liu, Y. et al. (2020) ‘A Secure Federated Transfer Learning Framework’, IEEE Intelligent Systems, 35(4), pp. 70–82. Available at: https://doi.org/10.1109/MIS.2020.2988525.

[4]: Pan, S.J. et al. (2010) ‘Cross-domain sentiment classification via spectral feature alignment’, in Proceedings of the 19th international conference on World wide web. WWW ’10: The 19th International World Wide Web Conference, Raleigh North Carolina USA: ACM, pp. 751–760. Available at: https://doi.org/10.1145/1772690.1772767.

[5]: Yang, Q. et al. (2019) ‘Federated Machine Learning: Concept and Applications’. arXiv. Available at: https://doi.org/10.48550/arXiv.1902.04885.

#### Literature providing attacks
 - In general, this categorization does not provide different types of attacks, as it is mainly about the data layer of the model, and not about its characteristics.
 - However, Transfer Learning has some attacks in literature, though *not in the federated setting.*
 - [6]: Wang, S. et al. (2022) ‘Backdoor Attacks Against Transfer Learning With Pre-Trained Deep Learning Models’, IEEE Transactions on Services Computing, 15(3), pp. 1526–1539. Available at: https://doi.org/10.1109/TSC.2020.3000900. (#TODO): Read in depth
