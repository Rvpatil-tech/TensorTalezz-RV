"""
Education Module
================

Interactive tools for teaching AI/ML concepts.

Functions:
    - explain: Analogy-driven explanations of AI/ML concepts.
    - visualize_algorithm: Step-by-step algorithm demonstration.
    - concept_demo: Toy-dataset demonstrations of ML ideas.
    - quiz: Generate short quizzes for learners.
    - compare_algorithms: Run & explain differences between algorithms.
"""

import random
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression, make_blobs
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ═══════════════════════════════════════════════════════════════════════════════
# Concept explanations (analogy-driven)
# ═══════════════════════════════════════════════════════════════════════════════

_EXPLANATIONS = {
    "overfitting": {
        "title": "Overfitting",
        "analogy": (
            "Imagine studying for a test by memorizing every single question in the "
            "practice book — word for word. You'd ace the practice book, but when the "
            "real test asks things slightly differently, you'd be lost. That's overfitting: "
            "the model learns the training data *too* well, including its noise and quirks, "
            "so it fails to generalize to new, unseen data."
        ),
        "formal": (
            "Overfitting occurs when a model captures noise in the training data "
            "rather than the underlying pattern. It results in low training error "
            "but high validation/test error."
        ),
        "tips": [
            "Use more training data.",
            "Apply regularization (L1/L2).",
            "Simplify the model (fewer parameters).",
            "Use cross-validation.",
            "Add dropout (in neural networks).",
        ],
    },
    "underfitting": {
        "title": "Underfitting",
        "analogy": (
            "Imagine trying to draw a detailed portrait using only a single straight line. "
            "No matter how hard you try, a line can't capture the complexity of a face. "
            "That's underfitting: the model is too simple to capture the patterns in the data."
        ),
        "formal": (
            "Underfitting occurs when a model is too simple to learn the underlying "
            "structure of the data. Both training and test errors are high."
        ),
        "tips": [
            "Use a more complex model.",
            "Add more features.",
            "Reduce regularization.",
            "Train for more epochs.",
        ],
    },
    "bias_variance": {
        "title": "Bias–Variance Tradeoff",
        "analogy": (
            "Think of throwing darts. High *bias* means you consistently hit the wrong "
            "spot (systematic error). High *variance* means your throws are scattered "
            "all over the board (inconsistent). The sweet spot is low bias AND low "
            "variance — consistently hitting the bullseye."
        ),
        "formal": (
            "Bias is the error from wrong assumptions; variance is sensitivity to "
            "small fluctuations in the training set. Total error = Bias² + Variance + "
            "Irreducible error. Reducing one often increases the other."
        ),
        "tips": [
            "Complex models → low bias, high variance.",
            "Simple models → high bias, low variance.",
            "Use ensemble methods to balance both.",
        ],
    },
    "gradient_descent": {
        "title": "Gradient Descent",
        "analogy": (
            "Imagine you're blindfolded on a hilly landscape and want to reach the "
            "lowest valley. You feel the slope under your feet and take a step downhill. "
            "You keep repeating this process — that's gradient descent. The *learning rate* "
            "is how big each step is: too big and you overshoot, too small and you crawl."
        ),
        "formal": (
            "Gradient descent is an iterative optimization algorithm that updates model "
            "parameters in the direction of the steepest decrease of the loss function. "
            "Update rule: θ = θ − α·∇L(θ), where α is the learning rate."
        ),
        "tips": [
            "Start with a moderate learning rate (e.g. 0.01).",
            "Use learning rate scheduling or adaptive optimizers (Adam, RMSprop).",
            "Mini-batch gradient descent balances speed and stability.",
        ],
    },
    "neural_network": {
        "title": "Neural Networks",
        "analogy": (
            "Think of a neural network like a factory assembly line. Raw materials "
            "(inputs) enter one end. At each station (layer), workers (neurons) transform "
            "the materials a little — bending, painting, assembling. By the time the product "
            "reaches the end, it's a finished item (prediction). Training is like improving "
            "the workers' skills through feedback from quality control."
        ),
        "formal": (
            "A neural network is a parameterized function composed of layers of linear "
            "transformations followed by non-linear activation functions. Trained via "
            "back-propagation and gradient descent."
        ),
        "tips": [
            "Start simple — a few layers, then scale up.",
            "Use ReLU activation for hidden layers as a default.",
            "Normalize inputs for faster convergence.",
        ],
    },
    "regularization": {
        "title": "Regularization",
        "analogy": (
            "Imagine you're packing for a trip and can only bring a small suitcase. "
            "You're forced to pick only the essentials. Regularization does the same "
            "for a model: it penalizes complexity, forcing it to keep only the most "
            "important features and reducing overfitting."
        ),
        "formal": (
            "Regularization adds a penalty term to the loss function to discourage "
            "overly complex models. L1 (Lasso) promotes sparsity; L2 (Ridge) shrinks "
            "weights towards zero."
        ),
        "tips": [
            "L1 for feature selection.",
            "L2 (Ridge) when all features are potentially useful.",
            "Elastic Net combines both L1 and L2.",
        ],
    },
    "cross_validation": {
        "title": "Cross-Validation",
        "analogy": (
            "Imagine you have 5 friends, and you want to test a recipe. Instead of "
            "always using the same friend as a taste-tester, you rotate: each friend "
            "gets a turn while the other 4 help you cook. That way, your recipe is "
            "tested by everyone. That's k-fold cross-validation."
        ),
        "formal": (
            "K-fold cross-validation splits data into k subsets. The model is trained "
            "k times, each time using k-1 folds for training and 1 for validation. "
            "The average score gives a robust performance estimate."
        ),
        "tips": [
            "k=5 or k=10 are common choices.",
            "Use stratified k-fold for imbalanced datasets.",
            "Leave-one-out CV is k=n (expensive but thorough).",
        ],
    },
    "decision_tree": {
        "title": "Decision Trees",
        "analogy": (
            "Think of playing 20 Questions. You ask yes/no questions to narrow down "
            "the answer. A decision tree works similarly — it asks a series of if/else "
            "questions about the features to arrive at a prediction."
        ),
        "formal": (
            "A decision tree recursively splits the feature space based on thresholds "
            "that maximize information gain (or minimize impurity). Leaf nodes contain "
            "the final predictions."
        ),
        "tips": [
            "Prone to overfitting — use pruning or max_depth.",
            "Ensemble methods (Random Forest, Gradient Boosting) improve performance.",
            "Great for interpretability.",
        ],
    },
    "random_forest": {
        "title": "Random Forest",
        "analogy": (
            "Imagine asking a crowd of people to vote on a decision. Each person has "
            "seen a slightly different subset of evidence. Individually they might be "
            "wrong, but the majority vote is usually right. A Random Forest is a crowd "
            "of decision trees, each trained on a random sample."
        ),
        "formal": (
            "Random Forest is an ensemble method that trains multiple decision trees "
            "on bootstrapped subsets of the data with random feature selection. "
            "Predictions are aggregated via majority vote (classification) or averaging "
            "(regression)."
        ),
        "tips": [
            "More trees generally = better, but with diminishing returns.",
            "Robust against overfitting compared to a single tree.",
            "Check feature_importances_ for interpretability.",
        ],
    },
    "svm": {
        "title": "Support Vector Machines (SVM)",
        "analogy": (
            "Imagine two groups of marbles on a table. You want to place a ruler "
            "(line) between them such that the gap on each side is as wide as possible. "
            "SVM finds that optimal separator — the *maximum margin hyperplane*. "
            "The marbles closest to the ruler are the *support vectors*."
        ),
        "formal": (
            "SVM finds the hyperplane that maximizes the margin between classes. "
            "The kernel trick enables non-linear decision boundaries by mapping "
            "data into higher-dimensional spaces."
        ),
        "tips": [
            "Works well in high-dimensional spaces.",
            "The 'C' parameter controls regularization.",
            "Common kernels: linear, RBF, polynomial.",
        ],
    },
    "linear_regression": {
        "title": "Linear Regression",
        "analogy": "Imagine trying to draw a single straight ruler line through a scatter of stars to show their general direction. It's about finding the underlying straight-line trend in messy data.",
        "formal": "A linear approach to modeling the relationship between a scalar response and one or more explanatory variables by fitting a linear equation to observed data.",
        "tips": ["Assumes linear relationship.", "Sensitive to outliers.", "Check homoscedasticity."],
    },
    "logistic_regression": {
        "title": "Logistic Regression",
        "analogy": "Think of an admissions officer giving you a probability (0% to 100%) of getting into college based on your grades. If it's over 50%, you're in (Class 1). Otherwise, you're out (Class 0).",
        "formal": "A statistical model that models the probability of a binary class using a logistic function.",
        "tips": ["Used for classification, not regression.", "Output is a probability.", "Assumes linear decision boundary."],
    },
    "knn": {
        "title": "K-Nearest Neighbors (KNN)",
        "analogy": "If you want to know if a neighborhood is wealthy, look at the 5 houses closest to the one you're interested in. You judge the unknown based on its closest neighbors.",
        "formal": "A non-parametric, lazy learning algorithm that classifies new cases based on a similarity measure (e.g., distance functions) to its 'k' nearest neighbors in the training data.",
        "tips": ["Requires feature scaling.", "Slow at inference time.", "Choose 'k' via cross-validation."],
    },
    "naive_bayes": {
        "title": "Naive Bayes",
        "analogy": "Imagine filtering spam: if an email says 'free money', it's probably spam. You 'naively' assume 'free' and 'money' are independent clues that add up to a final verdict.",
        "formal": "A family of simple probabilistic classifiers based on applying Bayes' theorem with strong (naive) independence assumptions between the features.",
        "tips": ["Works great for text classification.", "Fast to train.", "Assumes features are completely independent."],
    },
    "k_means_clustering": {
        "title": "K-Means Clustering",
        "analogy": "Imagine dropping 'k' magnets onto a table of iron filings. The filings snap to the closest magnet, forming 'clusters'. The magnets then move to the center of their clusters until nothing shifts.",
        "formal": "An unsupervised algorithm that partitions n observations into k clusters, where each observation belongs to the cluster with the nearest mean (centroid).",
        "tips": ["Must specify 'k' beforehand.", "Sensitive to initial centroid placement.", "Assumes spherical clusters."],
    },
    "hierarchical_clustering": {
        "title": "Hierarchical Clustering",
        "analogy": "Think of organizing files into folders. You group similar files into a subfolder, then group those subfolders into a larger folder, building a tree of relationships from the bottom up.",
        "formal": "An unsupervised method that builds a hierarchy of clusters either agglomerative (bottom-up) or divisive (top-down), typically visualized as a dendrogram.",
        "tips": ["No need to pre-specify the number of clusters.", "Computationally heavy for large datasets.", "Dendrograms are great but hard to read if too large."],
    },
    "backpropagation": {
        "title": "Backpropagation",
        "analogy": "Imagine a chef tasting a soup, realizing it needs salt, and shouting back to the kitchen to adjust the recipe. Backpropagation sends the 'error' backward through the network so it can adjust its weights.",
        "formal": "An algorithm used in training neural networks to calculate the gradient of the loss function with respect to the weights by applying the chain rule of calculus backward from output to input.",
        "tips": ["Essential for training deep networks.", "Prone to vanishing gradient problem.", "Relies heavily on differentiable activation functions."],
    },
    "learning_rate": {
        "title": "Learning Rate",
        "analogy": "When walking down a dark hill to find the bottom, taking tiny steps takes forever, but leaping might make you jump right past the valley. The learning rate is your step size.",
        "formal": "A hyperparameter that controls how much to change the model in response to the estimated error each time the model weights are updated.",
        "tips": ["Too high: model diverges or oscillates.", "Too low: training takes forever.", "Use learning rate schedulers to decrease it over time."],
    },
    "batch_vs_stochastic_gradient_descent": {
        "title": "Batch vs Stochastic Gradient Descent",
        "analogy": "Batch GD evaluates every student's homework before deciding how to teach the next lesson. Stochastic GD adjusts the lesson plan immediately after grading a single student's homework.",
        "formal": "Batch computes the gradient over the entire dataset. Stochastic (SGD) computes it per sample. Mini-batch is the common middle ground.",
        "tips": ["Mini-batch is the industry standard.", "SGD is faster per iteration but noisier.", "Batch is stable but memory-intensive."],
    },
    "activation_functions": {
        "title": "Activation Functions (ReLU, Sigmoid, Tanh)",
        "analogy": "Like a bouncer at a club. A neuron receives inputs, and the activation function decides if the signal is strong enough to mathematically 'let it through' to the next layer.",
        "formal": "Mathematical equations attached to each neuron in a network that determine whether it should be activated or not, introducing non-linearity to the model.",
        "tips": ["ReLU is the default for hidden layers.", "Sigmoid is for binary classification outputs.", "Softmax is for multi-class classification outputs."],
    },
    "loss_functions": {
        "title": "Loss Functions",
        "analogy": "Think of playing 'Hot and Cold'. The loss function is the person telling you how far away you are from the hidden object. The lower the loss, the 'hotter' (closer) you are.",
        "formal": "A method of evaluating how well your algorithm models your dataset. It measures the distance between the model's predictions and the actual true values.",
        "tips": ["MSE for regression tasks.", "Cross-Entropy for classification tasks.", "The goal of training is to minimize this expected loss."],
    },
    "cnns": {
        "title": "Convolutional Neural Networks (CNNs)",
        "analogy": "Imagine scanning a large painting with a tiny magnifying glass. You look for edges, then textures, then whole shapes. CNNs 'slide' filters over images to detect spatial patterns hierarchically.",
        "formal": "A class of deep neural networks, most commonly applied to analyzing visual imagery, using convolutional layers to capture spatial hierarchies of features.",
        "tips": ["State-of-the-art for image data.", "Involves Conv layers, Pooling layers, and Fully Connected layers.", "Reduces parameters drastically compared to dense networks."],
    },
    "rnns": {
        "title": "Recurrent Neural Networks (RNNs)",
        "analogy": "Reading a book requires remembering the previous sentence to understand the current one. RNNs have a 'loop' allowing information to persist, acting as an internal short-term memory.",
        "formal": "A class of artificial neural networks where connections between nodes form a directed graph along a temporal sequence, allowing it to exhibit temporal dynamic behavior.",
        "tips": ["Used for time-series and text.", "Prone to vanishing gradients (solving this gave us LSTMs/GRUs).", "Replaced largely by Transformers for NLP."],
    },
    "transformers": {
        "title": "Transformers & Attention Mechanism",
        "analogy": "Imagine reading a sentence and instantly knowing which earlier words are most important for context without having to read them sequentially. 'Attention' lets the model focus on what matters most simultaneously.",
        "formal": "A deep learning architecture relying entirely on self-attention mechanisms to compute representations of its input and output without using sequence-aligned RNNs or convolution.",
        "tips": ["Backbone of LLMs like GPT and BERT.", "Highly parallelizable training.", "Requires massive amounts of data and compute."],
    },
    "transfer_learning": {
        "title": "Transfer Learning",
        "analogy": "If you know how to ride a bicycle, learning to ride a motorcycle is much easier. Transfer learning takes a model trained on a massive task and fine-tunes it for a specific, smaller task.",
        "formal": "A research problem in machine learning that focuses on storing knowledge gained while solving one problem and applying it to a different but related problem.",
        "tips": ["Saves massive amounts of compute and time.", "Requires fewer data points.", "Common to freeze early layers and train only the head."],
    },
    "overparameterization": {
        "title": "Overparameterization",
        "analogy": "Imagine bringing a massive 1000-piece toolbox to fix a single leaky pipe. You have way more tools than you need, but studies show it somehow helps you find the right wrench faster.",
        "formal": "The phenomenon where deep learning models have far more parameters than training data points, yet still generalize well to unseen data without overfitting, contradicting classical ML theory.",
        "tips": ["Also known as the 'double descent' phenomenon.", "Requires powerful hardware.", "Empirically leads to smoother optimization landscapes."],
    },
    "confusion_matrix": {
        "title": "Confusion Matrix",
        "analogy": "Imagine a 2x2 grid showing: actual cats you called cats (True Positives), dogs you called cats (False Positives), actual dogs you called dogs (True Negatives), and cats you called dogs (False Negatives). It shows exactly how the model is 'confused'.",
        "formal": "A table layout that allows visualization of the performance of an algorithm. Each row represents instances in a predicted class, while each column represents instances in an actual class.",
        "tips": ["Vital for imbalanced datasets.", "Accuracy hides False Positives/Negatives; the matrix reveals them.", "Basis for Precision, Recall, and F1 scores."],
    },
    "roc_curve": {
        "title": "ROC Curve & AUC",
        "analogy": "Imagine tuning a metal detector. High sensitivity finds all coins (Recall) but also junk (False Positives). The ROC curve graphs this tradeoff at every possible sensitivity setting. AUC is your overall 'detector quality' score out of 1.0.",
        "formal": "A graphical plot (Receiver Operating Characteristic) illustrating the diagnostic ability of a binary classifier system as its discrimination threshold is varied. AUC is the area under this curve.",
        "tips": ["1.0 is perfect, 0.5 is random guessing.", "Excellent for comparing models.", "Independent of class distribution."],
    },
    "precision_vs_recall": {
        "title": "Precision vs Recall Tradeoff",
        "analogy": "Precision is 'When you cry wolf, is there actually a wolf?' Recall is 'Of all the actual wolves, did you cry wolf for every single one?' You often sacrifice one to improve the other.",
        "formal": "Precision is TP/(TP+FP), measuring exactness. Recall is TP/(TP+FN), measuring completeness. Increasing the threshold improves precision but lowers recall, and vice-versa.",
        "tips": ["High Precision for spam filters (don't block real mail).", "High Recall for cancer screening (don't miss a tumor).", "F1 Score balances both."],
    },
    "hyperparameter_tuning": {
        "title": "Hyperparameter Tuning",
        "analogy": "A model learns its weights, but you have to set the 'knobs' (hyperparameters) like learning rate or tree depth. Grid search is methodically trying every single knob combination; Random search is spinning knobs randomly until you find a good setting.",
        "formal": "The process of searching for the optimal configuration of hyperparameters (parameters whose values are set before the learning process begins) to optimize model performance.",
        "tips": ["Random search is often vastly more efficient than grid search.", "Always use a validation set.", "Bayesian optimization is a smarter, advanced alternative."],
    },
    "crossentropy_vs_accuracy": {
        "title": "Cross-Entropy vs Accuracy",
        "analogy": "Accuracy asks 'Did you get it right?' Cross-Entropy asks 'How confident were you when you got it right (or wrong)?' Being 51% confident and 99% confident both count as 'correct' in Accuracy, but Cross-Entropy rewards the 99% heavily.",
        "formal": "Accuracy is a hard metric (class match). Cross-entropy is a soft, continuous metric measuring the difference between two probability distributions (predicted vs actual).",
        "tips": ["Models optimize Cross-Entropy, not Accuracy.", "Always monitor both during training.", "Cross-Entropy heavily penalizes confident wrong answers."],
    },
    "ensemble_methods": {
        "title": "Ensemble Methods",
        "analogy": "A single investor might make a bad bet, but a diverse committee of 100 investors voting together rarely loses. Ensembling combines many weak learners to create one strong super-model.",
        "formal": "Machine learning techniques that combine several base models to produce one optimal predictive model.",
        "tips": ["Bagging (Random Forest) reduces variance.", "Boosting (XGBoost) reduces bias.", "Stacking combines different types of models using a meta-model."],
    },
    "gradient_boosting": {
        "title": "Gradient Boosting / XGBoost",
        "analogy": "Take a test, see what you got wrong. Give the mistakes to the next student to study specifically. They take a test, pass their mistakes to the next. The final score is everyone combined. Each tree fixes the errors of the previous one.",
        "formal": "A machine learning technique that produces a prediction model in the form of an ensemble of weak prediction models, typically decision trees. It builds the model in a stage-wise fashion.",
        "tips": ["Often beats Deep Learning on tabular data.", "XGBoost, LightGBM, and CatBoost are the industry standard implementations.", "Prone to overfitting if not tuned properly."],
    },
    "dimensionality_reduction": {
        "title": "Dimensionality Reduction",
        "analogy": "Imagine taking a 3D shadow of a complex object. You lose some details (the exact depth), but you can still completely recognize the object on a 2D piece of paper. You're compressing information while keeping the essence.",
        "formal": "The transformation of data from a high-dimensional space into a low-dimensional space so that the meaningful properties of the original data are retained.",
        "tips": ["PCA is linear and great for general compression.", "t-SNE/UMAP are non-linear and incredible for 2D/3D visualization.", "Helpful to defeat the 'Curse of Dimensionality'."],
    },
    "feature_engineering": {
        "title": "Feature Engineering",
        "analogy": "If you want to predict house prices, giving the ML model raw GPS coordinates is confusing. Giving it 'Distance to nearest hospital' is a brilliant insight. Feature engineering is giving the model pre-chewed food.",
        "formal": "The process of using domain knowledge to extract features (characteristics, properties, attributes) from raw data. These features can be used to improve the performance of machine learning algorithms.",
        "tips": ["Often yields higher ROI than hyperparameter tuning.", "Includes encoding, scaling, imputing, and creating interaction terms.", "Requires deep domain knowledge."],
    },
}


def explain(concept):
    """
    Provide an analogy-driven explanation of an AI/ML concept.

    Parameters
    ----------
    concept : str
        The concept to explain. Available concepts include: ``"overfitting"``,
        ``"underfitting"``, ``"bias_variance"``, ``"gradient_descent"``,
        ``"neural_network"``, ``"regularization"``, ``"cross_validation"``,
        ``"decision_tree"``, ``"random_forest"``, ``"svm"``.

    Returns
    -------
    info : dict
        Dictionary with ``title``, ``analogy``, ``formal``, and ``tips``.

    Example
    -------
    >>> info = explain("overfitting")
    """
    key = concept.lower().replace(" ", "_").replace("-", "_")
    if key not in _EXPLANATIONS:
        available = ", ".join(sorted(_EXPLANATIONS.keys()))
        print(f"❓ Unknown concept '{concept}'. Available: {available}")
        return None

    info = _EXPLANATIONS[key]
    print(f"\n{'='*60}")
    print(f"📖  {info['title']}")
    print(f"{'='*60}")
    print(f"\n🎯 Analogy:\n   {info['analogy']}")
    print(f"\n📐 Formal Definition:\n   {info['formal']}")
    print(f"\n💡 Tips:")
    for tip in info["tips"]:
        print(f"   • {tip}")
    print()
    return info


# ═══════════════════════════════════════════════════════════════════════════════
# Algorithm step-by-step visualization
# ═══════════════════════════════════════════════════════════════════════════════


def visualize_algorithm(algorithm="knn", n_samples=150, random_state=42, figsize=(10, 6)):
    """
    Demonstrate an algorithm step by step with visualizations.

    Currently supports:
        - ``"knn"``: K-Nearest Neighbors classification.
        - ``"linear_regression"``: Fitting a line to data.
        - ``"decision_tree"``: Splitting feature space.

    Parameters
    ----------
    algorithm : str, optional
        Algorithm to visualize. Default is ``"knn"``.
    n_samples : int, optional
        Number of synthetic data points. Default is ``150``.
    random_state : int, optional
        Seed for reproducibility. Default is ``42``.
    figsize : tuple, optional
        Figure size. Default is ``(10, 6)``.

    Returns
    -------
    None

    Example
    -------
    >>> visualize_algorithm("knn")
    """
    rng = np.random.RandomState(random_state)
    algo = algorithm.lower().replace(" ", "_")

    if algo == "knn":
        _viz_knn(n_samples, rng, figsize)
    elif algo == "linear_regression":
        _viz_linear_regression(n_samples, rng, figsize)
    elif algo == "decision_tree":
        _viz_decision_tree(n_samples, rng, figsize)
    else:
        print(f"❓ Visualization not available for '{algorithm}'. Available: knn, linear_regression, decision_tree")


def _viz_knn(n_samples, rng, figsize):
    """KNN step-by-step."""
    X, y = make_blobs(n_samples=n_samples, centers=3, random_state=rng.randint(10000))
    query = rng.randn(1, 2) * 2

    fig, axes = plt.subplots(1, 3, figsize=(figsize[0] * 1.5, figsize[1]))
    k_values = [1, 5, 15]

    for ax, k in zip(axes, k_values):
        model = KNeighborsClassifier(n_neighbors=k).fit(X, y)
        pred = model.predict(query)[0]

        # Decision boundary
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.2, cmap="Set3")
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap="Set1", s=20, edgecolors="k", linewidth=0.5)
        ax.scatter(query[0, 0], query[0, 1], c="gold", s=200, marker="*", edgecolors="k", zorder=5)
        ax.set_title(f"KNN (k={k}) → Class {pred}")
        ax.grid(alpha=0.2)

    plt.suptitle("KNN: Effect of k on Classification", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()
    print("⭐ The gold star is the query point. Notice how k affects the prediction and boundary smoothness.")


def _viz_linear_regression(n_samples, rng, figsize):
    """Linear regression step-by-step."""
    X = np.sort(rng.rand(n_samples, 1) * 10, axis=0)
    y = 2.5 * X.ravel() + rng.randn(n_samples) * 3 + 5

    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)

    fig, axes = plt.subplots(1, 3, figsize=(figsize[0] * 1.5, figsize[1]))

    # Step 1: Data
    axes[0].scatter(X, y, alpha=0.5, s=20, color="steelblue")
    axes[0].set_title("Step 1: Raw Data")
    axes[0].grid(alpha=0.3)

    # Step 2: Fit line
    axes[1].scatter(X, y, alpha=0.5, s=20, color="steelblue")
    axes[1].plot(X, y_pred, "r-", linewidth=2, label=f"y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}")
    axes[1].legend()
    axes[1].set_title("Step 2: Fit the Line")
    axes[1].grid(alpha=0.3)

    # Step 3: Residuals
    axes[2].scatter(X, y, alpha=0.5, s=20, color="steelblue")
    axes[2].plot(X, y_pred, "r-", linewidth=2)
    for xi, yi, ypi in zip(X.ravel(), y, y_pred):
        axes[2].plot([xi, xi], [yi, ypi], "g-", alpha=0.3)
    axes[2].set_title("Step 3: Residuals (errors)")
    axes[2].grid(alpha=0.3)

    plt.suptitle("Linear Regression: Step by Step", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()
    print(f"📏 Equation: y = {model.coef_[0]:.2f}·x + {model.intercept_:.2f}")


def _viz_decision_tree(n_samples, rng, figsize):
    """Decision tree depth comparison."""
    X, y = make_classification(n_samples=n_samples, n_features=2, n_redundant=0,
                                n_clusters_per_class=1, random_state=rng.randint(10000))

    fig, axes = plt.subplots(1, 3, figsize=(figsize[0] * 1.5, figsize[1]))
    depths = [1, 3, None]

    for ax, depth in zip(axes, depths):
        model = DecisionTreeClassifier(max_depth=depth, random_state=42).fit(X, y)
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=20, edgecolors="k", linewidth=0.5)
        label = f"Depth = {depth}" if depth else "No Limit"
        acc = accuracy_score(y, model.predict(X))
        ax.set_title(f"{label} (Acc: {acc:.2f})")
        ax.grid(alpha=0.2)

    plt.suptitle("Decision Tree: Effect of Max Depth", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()
    print("🌳 Deeper trees fit training data better but may overfit (see the rightmost panel).")


# ═══════════════════════════════════════════════════════════════════════════════
# Concept demos
# ═══════════════════════════════════════════════════════════════════════════════


def concept_demo(concept="overfitting", n_samples=200, random_state=42, figsize=(12, 5)):
    """
    Use toy datasets to illustrate ML ideas like overfitting and underfitting.

    Parameters
    ----------
    concept : str, optional
        ``"overfitting"``, ``"underfitting"``, or ``"bias_variance"``.
        Default is ``"overfitting"``.
    n_samples : int, optional
        Number of data points. Default is ``200``.
    random_state : int, optional
        Seed. Default is ``42``.
    figsize : tuple, optional
        Figure size. Default is ``(12, 5)``.

    Returns
    -------
    None

    Example
    -------
    >>> concept_demo("overfitting")
    """
    rng = np.random.RandomState(random_state)
    concept_key = concept.lower().replace(" ", "_").replace("-", "_")

    if concept_key in ("overfitting", "underfitting", "bias_variance"):
        _demo_fitting(concept_key, n_samples, rng, figsize)
    else:
        print(f"❓ Demo not available for '{concept}'. Available: overfitting, underfitting, bias_variance")


def _demo_fitting(concept, n_samples, rng, figsize):
    """Demonstrate over/underfitting with polynomial regression."""
    X = np.sort(rng.rand(n_samples) * 6 - 3)
    y_true = np.sin(X) + 0.5 * np.cos(2 * X)
    y = y_true + rng.randn(n_samples) * 0.3

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    degrees = [1, 4, 15]
    labels = ["Underfitting (degree 1)", "Good Fit (degree 4)", "Overfitting (degree 15)"]
    x_plot = np.linspace(X.min(), X.max(), 300)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    for ax, deg, label in zip(axes, degrees, labels):
        coeffs = np.polyfit(X_train, y_train, deg)
        poly = np.poly1d(coeffs)

        train_pred = poly(X_train)
        test_pred = poly(X_test)
        train_err = np.mean((y_train - train_pred) ** 2)
        test_err = np.mean((y_test - test_pred) ** 2)

        ax.scatter(X_train, y_train, s=10, alpha=0.5, label="Train")
        ax.scatter(X_test, y_test, s=10, alpha=0.5, label="Test", marker="x")
        ax.plot(x_plot, poly(x_plot), "r-", linewidth=2)
        ax.set_title(f"{label}\nTrain MSE: {train_err:.3f} | Test MSE: {test_err:.3f}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle(f"Concept Demo: {concept.replace('_', ' ').title()}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# Quiz generator
# ═══════════════════════════════════════════════════════════════════════════════

_QUIZ_BANK = [
    {
        "question": "What does 'overfitting' mean?",
        "options": [
            "A) The model is too simple to capture patterns.",
            "B) The model memorizes training data and fails on new data.",
            "C) The model has too few parameters.",
            "D) The training data is too large.",
        ],
        "answer": "B",
        "explanation": "Overfitting means the model learns noise in the training data rather than the true pattern.",
    },
    {
        "question": "What is the purpose of a validation set?",
        "options": [
            "A) To train the model.",
            "B) To tune hyperparameters and monitor overfitting.",
            "C) To deploy the model.",
            "D) To clean the data.",
        ],
        "answer": "B",
        "explanation": "A validation set helps you tune hyperparameters and detect overfitting before evaluating on the test set.",
    },
    {
        "question": "Which of these is an ensemble method?",
        "options": [
            "A) Linear Regression",
            "B) Random Forest",
            "C) Logistic Regression",
            "D) K-Means Clustering",
        ],
        "answer": "B",
        "explanation": "Random Forest combines many decision trees to make better predictions.",
    },
    {
        "question": "What does the learning rate control in gradient descent?",
        "options": [
            "A) The number of features.",
            "B) The size of each update step.",
            "C) The number of training epochs.",
            "D) The size of the dataset.",
        ],
        "answer": "B",
        "explanation": "The learning rate determines how big of a step we take towards the minimum of the loss function.",
    },
    {
        "question": "What is 'bias' in the bias-variance tradeoff?",
        "options": [
            "A) Error from sensitivity to small fluctuations.",
            "B) Error from wrong assumptions in the model.",
            "C) Error from too much training data.",
            "D) Error from missing features.",
        ],
        "answer": "B",
        "explanation": "Bias is the systematic error from overly simplistic assumptions in the learning algorithm.",
    },
    {
        "question": "What is regularization used for?",
        "options": [
            "A) To speed up training.",
            "B) To increase dataset size.",
            "C) To prevent overfitting by penalizing complexity.",
            "D) To remove features.",
        ],
        "answer": "C",
        "explanation": "Regularization adds a penalty to the loss function to prevent the model from becoming too complex.",
    },
    {
        "question": "In K-Nearest Neighbors, what happens when K is very large?",
        "options": [
            "A) The model overfits.",
            "B) The model underfits (boundary becomes too smooth).",
            "C) The model becomes faster.",
            "D) No change in performance.",
        ],
        "answer": "B",
        "explanation": "A very large K considers many neighbors, smoothing out the boundary and potentially underfitting.",
    },
    {
        "question": "What metric is best for imbalanced classification?",
        "options": [
            "A) Accuracy",
            "B) F1 Score",
            "C) Mean Squared Error",
            "D) R² Score",
        ],
        "answer": "B",
        "explanation": "F1 Score balances precision and recall, making it suitable for imbalanced datasets where accuracy can be misleading.",
    },
    {
        "question": "What does a confusion matrix show?",
        "options": [
            "A) Feature importances.",
            "B) True vs predicted labels breakdown.",
            "C) Training loss over time.",
            "D) Data distribution.",
        ],
        "answer": "B",
        "explanation": "A confusion matrix displays TP, FP, TN, FN — showing where the model gets confused.",
    },
    {
        "question": "What is the 'kernel trick' in SVMs?",
        "options": [
            "A) A way to speed up training.",
            "B) A technique to map data into a higher-dimensional space for linear separation.",
            "C) A pruning strategy.",
            "D) A type of gradient descent.",
        ],
        "answer": "B",
        "explanation": "The kernel trick implicitly maps data into a higher-dimensional space where it becomes linearly separable.",
    },
]


def quiz(n_questions=5, shuffle=True):
    """
    Generate a short interactive quiz for learners.

    Parameters
    ----------
    n_questions : int, optional
        Number of questions. Default is ``5``.
    shuffle : bool, optional
        Randomize question order. Default is ``True``.

    Returns
    -------
    score : int
        Number of correct answers.

    Example
    -------
    >>> score = quiz(n_questions=3)
    """
    questions = _QUIZ_BANK.copy()
    if shuffle:
        random.shuffle(questions)
    questions = questions[:n_questions]

    print(f"\n{'='*60}")
    print(f"🧠  AI/ML Quiz — {n_questions} Questions")
    print(f"{'='*60}\n")

    score = 0
    for i, q in enumerate(questions, 1):
        print(f"Q{i}. {q['question']}")
        for opt in q["options"]:
            print(f"   {opt}")
        answer = input("   Your answer (A/B/C/D): ").strip().upper()

        if answer == q["answer"]:
            score += 1
            print(f"   ✅ Correct! {q['explanation']}\n")
        else:
            print(f"   ❌ Wrong. The answer is {q['answer']}. {q['explanation']}\n")

    pct = score / n_questions * 100
    print(f"{'='*60}")
    print(f"📊 Score: {score}/{n_questions} ({pct:.0f}%)")
    if pct == 100:
        print("🏆 Perfect score! You're an AI/ML star!")
    elif pct >= 70:
        print("👍 Great job! Keep learning!")
    else:
        print("📚 Keep studying — you'll get there!")
    print(f"{'='*60}\n")

    return score


# ═══════════════════════════════════════════════════════════════════════════════
# Compare algorithms
# ═══════════════════════════════════════════════════════════════════════════════


def compare_algorithms(
    algorithms=None,
    n_samples=300,
    task="classification",
    random_state=42,
    figsize=(14, 5),
):
    """
    Train multiple algorithms on a synthetic dataset and compare results.

    Parameters
    ----------
    algorithms : list of str or None, optional
        Algorithms to compare. If ``None``, defaults to a curated set.
    n_samples : int, optional
        Number of synthetic samples. Default is ``300``.
    task : str, optional
        ``"classification"`` (default) or ``"regression"``.
    random_state : int, optional
        Seed. Default is ``42``.
    figsize : tuple, optional
        Figure size. Default is ``(14, 5)``.

    Returns
    -------
    results : dict
        Per-algorithm metrics dictionary.

    Example
    -------
    >>> results = compare_algorithms()
    """
    rng = np.random.RandomState(random_state)

    if task == "classification":
        X, y = make_classification(
            n_samples=n_samples, n_features=2, n_redundant=0,
            n_clusters_per_class=1, random_state=random_state,
        )
        if algorithms is None:
            algorithms = ["Logistic Regression", "KNN", "Decision Tree", "Random Forest", "SVM"]

        algo_map = {
            "logistic regression": LogisticRegression(max_iter=1000),
            "knn": KNeighborsClassifier(),
            "decision tree": DecisionTreeClassifier(random_state=42),
            "random forest": RandomForestClassifier(random_state=42),
            "svm": SVC(random_state=42),
            "gradient boosting": GradientBoostingClassifier(random_state=42),
        }
    else:
        X, y = make_regression(n_samples=n_samples, n_features=2, noise=15, random_state=random_state)
        if algorithms is None:
            algorithms = ["Linear Regression", "Decision Tree", "Random Forest"]
        algo_map = {
            "linear regression": LinearRegression(),
            "decision tree": DecisionTreeRegressor(random_state=42),
            "random forest": RandomForestRegressor(random_state=42),
        }

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)

    results = {}
    fig, axes = plt.subplots(1, len(algorithms), figsize=figsize)
    if len(algorithms) == 1:
        axes = [axes]

    for ax, name in zip(axes, algorithms):
        key = name.lower()
        if key not in algo_map:
            print(f"⚠️  Unknown algorithm '{name}', skipping.")
            continue

        model = algo_map[key]
        t0 = time.time()
        model.fit(X_train, y_train)
        fit_time = time.time() - t0
        preds = model.predict(X_test)

        if task == "classification":
            acc = accuracy_score(y_test, preds)
            results[name] = {"accuracy": acc, "time": fit_time}

            # Decision boundary
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
            ax.contourf(xx, yy, Z, alpha=0.25, cmap="coolwarm")
            ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="coolwarm", s=15, edgecolors="k", linewidth=0.3)
            ax.set_title(f"{name}\nAcc: {acc:.2f} | {fit_time:.3f}s")
        else:
            from sklearn.metrics import r2_score as r2s
            r2 = r2s(y_test, preds)
            results[name] = {"r2": r2, "time": fit_time}
            ax.scatter(y_test, preds, alpha=0.5, s=15)
            mn, mx = min(y_test.min(), preds.min()), max(y_test.max(), preds.max())
            ax.plot([mn, mx], [mn, mx], "r--")
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title(f"{name}\nR²: {r2:.2f} | {fit_time:.3f}s")

        ax.grid(alpha=0.2)

    plt.suptitle(f"Algorithm Comparison ({task.title()})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()

    # Summary table
    print(f"\n{'Algorithm':<25}{'Score':<15}{'Time (s)':<10}")
    print("-" * 50)
    for name, res in results.items():
        score = res.get("accuracy", res.get("r2", 0))
        print(f"{name:<25}{score:<15.4f}{res['time']:<10.4f}")

    return results
