# Generative Deep Learning

This repository contains notes and code implementations for the book,
[Generative Deep Learning](https://www.amazon.com/Generative-Deep-Learning-Teaching-Machines/dp/1492041947).

The Jupyter notebooks in this repository are meant for use with
[Google Colab](https://colab.research.google.com/).

## Topics Covered

Below is a list of the topics covered in each notebook in this repository listed in the order you should read them.

### Chapter 1: Generative Models

#### `GenerativeModels.ipynb`

- Generative vs. discriminative modeling
- A framework for generative modeling
- Sample space
- Probability density function
- Parametric modeling
- Likelihood
- Maximum likelihood estimation (MLE)
- Multinomial distribution
- Naive Bayes assumption

### Chapter 2: Deep Learning

#### `DeepNeuralNetworks.ipynb`

- Deep neural networks using Keras
- Keras' sequential and functional APIs
- The `Input`, `Dense`, and `Flatten` layers
- Keras `Model` class
- Rectified linear units (ReLU) function
- Leaky ReLU
- Sigmoid function
- Softmax function
- `Adam` optimizer
- Categorical crossentropy
- Mean squared error
- Binary crossentropy
- Training models using Keras
- Making predictions with a Keras model
- Convolutional layers
- Convolutional neural networks (CNNs)
- Batch normalization
- `Dropout` layers
- Evaluating model performance

### Chapter 3: Variational Autoencoders

#### `Autoencoders.ipynb`

- Autoenocders
- Encoder and decoder networks
- Latent space
- Implementing an autoencoder in Keras
- MNIST dataset
- Training an autoencoder
- Analyzing the performance of an autoencoder
- Reconstructing images from the latent space
- Challenges of generating new images with an encoder

#### `VariationalAutoencoder.ipynb`

- Variational autoencoder (VAE)
- Multidimensional normal distribution
- Kullback Leibler (KL) divergence
- Implementing a VAE with Keras
- Reconstructing MNIST digits with a VAE
- Plotting the latent space
- Generating new images of MNIST digits using a VAE
- CelebA dataset
- Reconstructing pictures of CelebA faces with a VAE
- Generating new pictures of faces with a VAE
- Extracting feature vectors from the latent space
- Using feature vectors to change the output of a generative model
- Morphing between faces using a VAE

### Chapter 4: Generative Adversarial Networks

#### `GenerativeAdversarialNetwork.ipynb`

- Generative adversarial networks (GANs)
- Generator and discriminator networks
- Google's Quick, Draw! dataset
- Implementing a deep convolutional GAN (DCGAN) with Keras
- Training a GAN with Keras
- Generating new samples using a GAN

#### `WassersteinGAN.ipynb`

- Loss metrics in GANs
- Wasserstein loss
- Lipschitz constraint
- Wasserstein GAN (WGAN)
- Implementing a WGAN with Keras
- CIFAR18 dataset
- Training a WGAN
- Generating images of horses with the WGAN

#### `WGAN-GP.ipynb`

- Wasserstein GAN-Gradient Penalty (WGAN-GP)
- Gradient penalty loss
- Computing gradient penalty loss
- L2 (Euclidean) norm
- Implementing a WGAN-GP with Keras
- Training with a dataset larger than your available RAM
- Generating faces with a WGAN-GP
- Visualizing the training process with output at various stages of training

### Chapter 5: Paint

#### `CycleGAN.ipynb`

- Cycle-consistent adversarial networks (CycleGAN)
- CycleGAN generators
- U-Net
- ResNet
- CycleGAN loss functions
- CycleGAN discriminators
- Implementing a CycleGAN with Keras
- Changing pictures of apples to oranges (and vice versa)
- CycleGAN training sets
- apple2orange dataset
- Evaluating a CycleGAN against the 3 loss metrics
- Painting like Monet
- monet2photo dataset
- Turning photographs into paintings using CycleGAN

#### `NeuralStyleTransfer.ipynb`

- Neural style transfer
- VGG-19 network
- ImageNet dataset
- Content loss
- Style loss
- Total variance loss
- Implementing neural style transfer with Keras
- Running neural style transfer with a given input

### Chapter 6: Write

#### `LSTM.ipynb`

- Long short-term memory (LSTM) networks
- Recurrent neural networks (RNNs)
- LSTM cell
- Tokenizing text data
- Building a text dataset for training an LSTM
- `Embedding` layer
- Building an LSTM with Keras
- Stacked LSTMs
- Generating new text with an LSTM

#### `QAGenerator.ipynb`

- TensorFlow's qgen-workshop
- Encoder-decoder networks
- GRU cells
- Maluuba News QA repository
- Downloading CNN news stories
- Using the Maluuba News QA code to generate a training set
- Global Vector (GloVe) embeddings
- Downloading and using the embeddings for transfer learning
- Preprocessing and tokenizing news story data
- `Bidirectional` layer
- Implementing a QA generator with Keras
- Training the QA generator
- Using the model to plot the probability a word in a document is an answer to a question
- Generating new questions using the model

### Chapter 7: Compose

#### `Notation.ipynb`

- Midi files
- Bach's Cello Suites as midi files
- Converting midi files to wav files with Fluidsynth
- Playing music in Colab
- Musical notation
- music21 library
- Extracting sequential data from midi files with music21

#### `AttentionMechanism.ipynb`

- Attention mechanisms
- Extracting sequential data from Bach's Cello Suites
- Implementing an attention mechanism in Keras for an LSTM network
- Training an LSTM with attention
- Generating new music using a neural network

#### `MuseGAN.ipynb`

- MuseGAN
- JSB-Chorales dataset
- Converting sequential data into midi files
- Implementing MuseGAN with Keras
- Training MuseGAN
- Generating new music with MuseGAN
- Changing the input to MuseGAN

### Chapter 8: Play

#### `ReinforcementLearning.ipynb`

- Reinforcement learning (RL)
- RL environments
- RL agents
- Game state
- Actions and rewards
- Episode a.ka. rollout
- OpenAI Gym
- CarRacing-v0 environment
- Making observations in an OpenAI Gym environment
- Action spaces
- Plotting an animation of an agent in an environment

#### `WorldModel.ipynb`

- World model
- Dream environments
- Mixture density network (MDN)
- MDN-RNN
- Generating rollout data
- Training a VAE to recreate observations from an OpenAI Gym environment
- Implementing an MDN-RNN with Keras
- Training an MDN-RNN to predict the next game state
- Implementing an agent controller
- Creating a dream environment for an agent to learn car racing
- Covariance matrix adapation evolution strategy (CMA-ES)
- Training an agent in the dream environment
- Analyzing the agent's performance in the real environment

### Chapter 9: The Future of Generative Modeling

#### `FutureGenerativeModeling.ipynb`

- Transformer models
- Positional encoding
- Multihead attention
- Scaled dot-product attention
- TensorFlow notebook for using a trained transformer
- Bidirectional Encoder Representation from Transformers (BERT)
- GPT-2
- MuseNet
- Sparse Transformer model
- ProGAN
- Self-Attention GAN (SAGAN)
- BigGAN
- Adaptive instance normalization

## Papers cited

- [Martin Arjovsky, Soumith Chintala, and Léon Bottou, _Wasserstein GAN_](https://arxiv.org/pdf/1701.07875.pdf)
- [Andrew Brock, Jeff Donahue, Karen Simonyan, _Large Scale GAN Training for High Fidelity Natural Image Synthesis_](https://arxiv.org/abs/1809.11096)
- [R.H. Byrd, S.L. Hansen, J. Nocedal, Y.Singer, _A Stochastic Quasi-Newton Method for Large-Scale Optimization_](https://arxiv.org/abs/1401.7020)
- [Rewon Child, Scott Gray, Alec Radford, Ilya Sutskever, _Generating Long Sequences with Sparse Transformers_](https://arxiv.org/abs/1904.10509)
- [Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova, _BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding_](https://arxiv.org/abs/1810.04805)
- [Hao-Wen Dong, Wen-Yi Hsiao, Li-Chia Yang, Yi-Hsuan Yang, _MuseGAN: Multi-track Sequential Generative Adversarial Networks for Symbolic Music Generation and Accompaniment_](https://arxiv.org/abs/1709.06298)
- [Ian Goodfellow, _Generative Adversarial Networks_](https://channel9.msdn.com/Events/Neural-Information-Processing-Systems-Conference/Neural-Information-Processing-Systems-Conference-NIPS-2016/Generative-Adversarial-Networks)
- [Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, Aaron Courville, _Improved Training of Wasserstein GANs_](https://arxiv.org/abs/1704.00028)
- [David Ha, Jürgen Schmidhuber, _World Models_](https://arxiv.org/abs/1803.10122)
- [Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, _Deep Residual Learning for Image Recognition_](https://arxiv.org/pdf/1512.03385.pdf)
- [Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A. Efros, _Image-to-Image Translation with Conditional Adversarial Networks_](https://arxiv.org/pdf/1611.07004.pdf)
- [Tero Karras, Timo Aila, Samuli Laine, Jaakko Lehtinen, _Progressive Growing of GANs for Improved Quality, Stability, and Variation_](https://arxiv.org/abs/1710.10196)
- [Tero Karras, Samuli Laine, Timo Aila, _A Style-Based Generator Architecture for Generative Adversarial Networks_](https://arxiv.org/abs/1812.04948)
- [Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, _Language Models are Unsupervised Multitask Learners_](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Olaf Ronneberger, Philipp Fischer, and Thomas Brox, _U-Net: Convolutional Networks for Biomedical Image Segmentation_](https://arxiv.org/pdf/1505.04597.pdf)
- [Dmitry Ulyanov, Andrea Vedaldi, Victor Lempitsky, _Instance Normalization: The Missing Ingredient for Fast Stylization_](https://arxiv.org/pdf/1607.08022.pdf)
- [Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin, _Attention Is All You Need_](https://arxiv.org/abs/1706.03762)
- [Han Zhang, Ian Goodfellow, Dimitris Metaxas, Augustus Odena, _Self-Attention Generative Adversarial Networks_](https://arxiv.org/abs/1805.08318)
- [Jun-Yan Zhu, Taesung Park, Phillip Isola Alexei A. Efros, _Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks_](https://arxiv.org/pdf/1703.10593.pdf)
