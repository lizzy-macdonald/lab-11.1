In this assignment, you will develop an Artificial Neural Network (ANN) using deep learning techniques to predict the maximum load capacity (in tons) of bridges. You will:

Develop an ANN model using either TensorFlow or PyTorch.

Train your model on the provided dataset.

Evaluate your model’s performance and plot training/validation loss vs. epochs.

Include techniques such as early stopping, L2 regularization, and dropout.

Save your preprocessing pipeline as a .pkl file so that you can deploy your model consistently.

Deploy your solution on GitHub and build an interactive Streamlit app.

Dataset Description
Please note the dataset used in this lab assignment was created for education purpose ONLY. They may not represent actual experimental data.

The dataset, lab_11_bridge_data.csv, contains the following columns:

Bridge_ID: Unique identifier for each bridge.
Span_ft: The length of the bridge span in feet.
Deck_Width_ft: The width of the bridge deck in feet.
Age_Years: Age of the bridge in years.
Num_Lanes: Number of lanes on the bridge.
Material: Categorical variable indicating the primary construction material (e.g., “Steel”, “Concrete”, “Composite”).
Condition_Rating: An index (from 1 to 5) representing the overall condition (with 5 being excellent, this is NOT the rating being used by National Bridge Inventory)
Max_Load_Tons: The maximum load capacity of the bridge in tons (this is the target variable).
A sample of the dataset (first five rows) is shown below:

Bridge_ID	Span_ft	Deck_Width_ft	Age_Years	Num_Lanes	Material	Condition_Rating	Max_Load_Tons
B001	250	40	20	4	Steel	4	180
B002	300	45	35	2	Concrete	3	150
B003	150	30	15	2	Composite	5	120
(The full dataset is provided as lab_11_bridge_data.csv.)

Assignment Guidelines
Deliverables
Data Exploration and Preprocessing:

Load and explore the dataset.
Handle missing values, perform encoding on the categorical variable, and normalize/standardize features as needed.
Model Development:

You may use either Tensorflow or PyTorch (our previous examples such as housing price was created using Tensorflow though)
Develop an ANN to predict Max_Load_Tons.
Practicing applying the techniques (early stopping, L2 weight regularization, dropout layers) we used in lecture 11.1 to build a more robust ANN model
Training and Evaluation:

Split the data into training and test sets.
Train your model and evaluate its performance by plotting training/validation loss v.s. epochs
Save your model files at the end of your scripts for deployment: If using Tensorflow: model.save(“tf_bridge_model.h5”)
Deployment:

Upload your code to GitHub.
Create a Streamlit app that allows users to input new bridge data and see the predicted maximum load.
Submission:

Copy and paste your web app address in the assignment comments area.
Python scripts used for training and evaluation
Python scripts used for web app deployment
