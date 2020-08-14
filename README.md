# Encoder Decoder model deployed into a web app

This is my implementation of the encoder decoder RNN architecture proposed [by this paper (Cho et al)](https://arxiv.org/abs/1406.1078) in 2014. The model was trained using Keras on real sales data, then exported as a *.keras file later to be deployed on a Flask Web App. To build the dashboard, I used this [Colorlib template](https://github.com/puikinsh/Adminator-admin-dashboard).

The Notebook where the model training took place is included in the project (`model.ipynb`) . It is structured into three main parts. Retrieving / Normalizing / Reshaping the data, and building the model using Keras functional API then testing and exporting the model.


In order to get the utmost accuracy, I used [Scikit Optimize](https://scikit-optimize.github.io/stable/) to optimize the hyperparameters of the model. It uses a technique called Bayesian Optimization, which is not a gradient based optimization. This great [article](https://distill.pub/2020/bayesian-optimization/) on distill explains what happens well. Training this meta-optimizer took around a day and a half. The code for that exists in the `meta.py` file.

After training, the meta-optimizer will output the best possible combination of hyperparameters with their loss, export the best performing model as *.keras, and serialize the scitkit optimize object for further heuristics on the training. For more, read the code or refer to the ScikOpt [documentation](https://scikit-optimize.github.io/stable/modules/plots.html). To change the model the dashboard uses, simply replace the old model with the new one (`model.keras`) in `static/model` directory.

This project was my graduation project.


##
The implementation is pretty straightforward; for any questions however, contact me: 

* Linkedin: [Abdessalem Boukil](https://www.linkedin.com/in/abdessalem-boukil-37923637/)
* E-mail: boukil98@gmail.com


