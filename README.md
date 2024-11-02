This project implements a **Top-N recommendation system** using **heterogeneous Graph Neural Networks (GNNs)** to address the cold-start problem in the context of e-commerce. The system is designed for the "Informatique" (computers and electronics) category on the Ouedkniss online marketplace, but the approach is generalizable to other e-commerce platforms.

## Problem Statement

Traditional recommendation methods, such as collaborative filtering, struggle with the cold-start problem (new users/items) and often fail to capture complex relationships between products, users, and other relevant factors. This project aims to overcome these limitations with a GNN-based approach.

## Solution

This project uses a heterogeneous GNN model to learn from a graph representing user-product interactions, product subcategories, user preferences, and geographic locations. The model is trained using **Bayesian Personalized Ranking (BPR) loss** to optimize for ranking recommendations.

---

## Table of Contents

1. [Introduction to Traditional Recommendation Systems](#introduction-to-traditional-recommendation-systems)
2. [Proposed Approach](#proposed-approach)
   - [Advantages of GNNs in Recommendations](#advantages-of-gnns-in-recommendations)
   - [Product and User Features](#product-and-user-features)
3. [Model Implementation](#model-implementation)
4. [Key Features](#key-features)
5. [Dataset](#dataset)
6. [Requirements](#requirements)

---

## Introduction to Traditional Recommendation Systems

Traditional recommendation systems typically fall into three main categories:

- **Content-Based Filtering:** Recommends items similar to those a user has liked in the past, relying on item feature similarity. For example, if a user has browsed laptops, a content-based system might recommend similar laptops based on shared specifications.
- **Collaborative Filtering:** Predicts user preferences based on the preferences of similar users, but suffers from the cold-start problem and struggles with sparse datasets.
- **Hybrid Approaches:** Combine content-based and collaborative filtering to leverage both strengths, such as switching methods based on data availability.

However, these methods struggle in complex e-commerce environments with diverse factors like product categories, user location, and delivery options. This complexity, particularly evident in Ouedkniss’s "Informatique" category, makes traditional methods less effective.

## Proposed Approach

### Advantages of GNNs in Recommendations

Graph Neural Networks (GNNs) model complex relationships within e-commerce data by representing it as a graph. Unlike traditional methods, GNNs can capture the interconnected nature of user preferences, product attributes, and other relevant factors.

In this project, we use a **heterogeneous GNN** to represent different types of nodes (users, products) and edges (purchases, subcategory memberships). This approach allows the model to capture nuances, such as location-based preferences and category-specific interests. Furthermore, GNNs mitigate the cold-start problem by leveraging information from related items and users, even with sparse direct interaction data.

### Product and User Features

#### Product Features
| **Feature**            | **Description**                                        |
|------------------------|--------------------------------------------------------|
| Product Name           | Name of the product                                    |
| Wilaya                 | Algerian province                                      |
| Time Since Uploading   | Time elapsed since the product listing was posted      |
| Delivery Type          | Delivery options offered by the seller                 |
| Store Name             | Name of the store offering the product (if applicable) |
| Product Type           | Whether the product is new or used                     |
| Subcategory            | Subcategory of the product within "Informatique"       |
| Price                  | Price of the product                                   |

#### User Features
| Feature             | Description                                                 |
|----------------------|-------------------------------------------------------------|
| Wilaya              | User's preferred purchase location                           |
| Preferred Subcategory | Subcategory preferred by the user                         |
| Purchased Products  | Products previously purchased by the user                   |
| Rating              | User ratings for purchased products                         |

Synthetic purchase histories were generated to reflect realistic user behavior, resulting in 1750 entries in the user-product interaction dataset.

---

## Model Implementation

### BPR Model Enrollment Process

The **BPR (Bayesian Personalized Ranking)** model ranks items by learning user preferences through positive and negative interactions. This personalizes item ranking based on user-specific interactions.

### BPR Model Definition

1. **BPRModel Class**: Extends `torch.nn.Module` and encapsulates the BPR model structure.
   - **Initialization**: User and product embeddings (`nn.Embedding`) allow learning of latent features.
   - **Heterogeneous Graph Convolution**: Defined using `HeteroConv` with `GATConv` to handle different relationships (e.g., purchases, preferences).
2. **Forward Pass**: Applies convolution layers to generate updated embeddings.
3. **Embedding Retrieval**: Retrieves embeddings for specific user and product IDs.
4. **Loss Calculation**: Computes BPR loss, focusing on positive vs. negative interactions for personalized ranking.

---

## Key Features

- **Heterogeneous Graph**: Represents relationships, including user-product purchases, subcategory memberships, and location preferences.
- **Cold-Start Mitigation**: Leverages graph structure to recommend new items based on related products and users.
- **Evaluation with MAP@5**: Evaluates model performance with Mean Average Precision at 5 (MAP@5) to assess Top-N recommendations.
- **Early Stopping**: Prevents overfitting, ensuring optimal model performance.
- **Visualization**: Provides a function to visualize the graph structure for analysis and understanding.

---

## Dataset

The dataset consists of scraped product listings from the "Informatique" category on Ouedkniss, with synthetic user data (preferences and purchase histories) generated to facilitate training and evaluation.

---

## Requirements

To install dependencies:

```bash
pip install torch torch-geometric tqdm scikit-learn networkx matplotlib pandas numpy
