# Neural-Transfer

## Comparison of Different Neural Network Architectures for Neural Transfer Learning

This is the codes for GR5242 Final Project.

We built a neural transfer system and compared the performance of this system under different hyper-parameter settings as well as different neural networks architecture. We compared the results between different neural networks architectures including VGG-19, ResNet50, Xception in our neural transfer system. 

The hyper-parameters we studied in this projects are weights of loss \& style functions($\alpha$ and $\beta$), noise ratio for the content pictures($\theta$), learning rate for the gradient descent optimizer($\eta$, we used Stochastic gradient descent as optimizer) We also observed the results under different iterations and random seeds. 


```
./
├── README.md
├── contents
│   └── zelda.jpg
├── main.ipynb
├── outputs
│   ├── 0.jpg
│   ├── 10.jpg
│   ├── 100.jpg
│   ├── 1000.jpg
│   ├── 110.jpg
│   ├── 120.jpg
│   ├── 1200.jpg
│   ├── 130.jpg
│   ├── 140.jpg
│   ├── 1400.jpg
│   ├── 150.jpg
│   ├── 160.jpg
│   ├── 1600.jpg
│   ├── 170.jpg
│   ├── 180.jpg
│   ├── 1800.jpg
│   ├── 190.jpg
│   ├── 1999.jpg
│   ├── 20.jpg
│   ├── 200.jpg
│   ├── 30.jpg
│   ├── 40.jpg
│   ├── 400.jpg
│   ├── 50.jpg
│   ├── 60.jpg
│   ├── 600.jpg
│   ├── 70.jpg
│   ├── 80.jpg
│   ├── 800.jpg
│   └── 90.jpg
├── styles
│   └── kanagawa.jpeg
└── utils
    └── model.py

4 directories, 35 files
```
