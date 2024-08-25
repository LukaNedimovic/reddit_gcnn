<h1>📝 Reddit GCNN - Implementation of Graph Convolutional Neural Networks in Graph Classification problem </h1>

<b>Reddit GCNN</b> is a project created for the course <b>"Experiments with Neural Networks 1"</b> (Faculty of Sciences, University of Novi Sad) <br/>
Date of creation: <b>July, 2024</b>. <br/>

<h2> ℹ️ General Information </h2>
<a href="https://huggingface.co/datasets/graphs-datasets/reddit_threads"><b>Reddit threads dataset</b></a> used forms a graph of user interactions, out of which GCN needs to learn useful features and predict whether given interactions are part of a <b>discussion</b> (label <code>1</code>) or not (label <code>0</code>). 

This project uses <b>Graph Convolutional Neural Networks</b>, experimenting with <code>pytorch_geometric</code> convolutional layers - <code>GCNConv</code> and <code>SAGEConv</code> layers, showcasing that, under similar hyperparameters (e.g. <b>embedding dimensions</b>) these two may produce similar results. We implement a straightforward structure - several layers of <b>graph convolution</b>, followed by several layers of <b>multi-layer perceptron</b>. The output of the MLP is passed through the sigmoid function to generate prediction in range <code>(0, 1)</code>.
GCN is used to learn node embeddings, which are shared among present graphs. For the simplicity of implementation, we resort ourselves to <code>1000</code> entries.

<h2> 🚀 Quick Start </h2>
<pre>
<code>git clone https://github.com/LukaNedimovic/reddit_gcnn.git
cd reddit_gcnn
source ./setup.sh</code></pre>

<h2> 📁 Folder Tree </h2>
<pre>
  <code>
pmf_exp_nn_1_proj  
├── data
│   ├── plots               # Grouped and separated accuracy plots                 
│   ├── load_data.py        # Load Reddit dataset
│   └── training_data.csv   # General data collected after training
├── model
│   └── gcn_model.py        # General implementation of Graph Convolutional Network model
├── README.md               # You are reading this!                          
├── scripts
│   ├── plot                # Plot train / test accuracies post-training
│   ├── load                # Exemplary script that can be used to load a model
│   ├── train               # Scripts used for training respective model architectures
│   │   ├── gcn_s
│   │   ├── gcn_m
│   │   ├── gcn_l           
│   │   ├── sage_s
│   │   ├── sage_m                 
│   │   └── sage_l
│   └── update_req.sh       # Update requirements after changes
├── train
│   └── trainer.py          # File containing training method and configuration
├── utils                   # Generally useful functionalities
│    ├── argparser.py       # Parse cmdline arguments
│    ├── const.py           # Constants useful throughout training process
│    ├── plot.py            # Plot nice matplotlib graphs
│    └── log.py             # Making output look pleasant
├── main.py                 # Training starts here                 
└── setup.py                # Set up the environment
  </code>
</pre>


<h2> 🔥 Train a Model </h2>
To start the training process, run one of the scripts present in the <code>./scripts/train/</code> folder.
