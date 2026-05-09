"""
Shadow - Differentiable K-means for Soft Clustering
PyTorch-based soft K-means that maintains gradient flow for meta-learning.

Why Differentiable K-means?
- Soft assignments: Probability distributions over clusters (not hard assignments)
- Gradient flow: Can be optimized with backpropagation
- Meta-learning ready: Enables future meta-learning router (Phase 3)
- Better than GMM: Simpler, more interpretable, still differentiable
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
from tqdm import tqdm


class DifferentiableKMeans(nn.Module):
    """
    Differentiable K-means clustering with soft assignments.
    Uses temperature-scaled softmax for soft cluster assignments.
    """
    
    def __init__(
        self,
        n_clusters: int = 10,
        n_features: int = 50,
        temperature: float = 1.0,
        init_method: str = "kmeans++"
    ):
        """
        Initialize Differentiable K-means.
        
        Args:
            n_clusters: Number of clusters
            n_features: Number of features (from Sparse PCA)
            temperature: Temperature for softmax (lower = harder assignments)
            init_method: "kmeans++" or "random"
        """
        super().__init__()
        self.n_clusters = n_clusters
        self.n_features = n_features
        self.temperature = temperature
        
        # Cluster centers (learnable parameters)
        self.cluster_centers = nn.Parameter(torch.randn(n_clusters, n_features))
        
        self.init_method = init_method
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute soft cluster assignments.
        
        Args:
            X: Input features (batch_size, n_features)
        
        Returns:
            Soft assignments (batch_size, n_clusters) - probability distributions
        """
        # Compute distances to cluster centers
        # X: (batch_size, n_features)
        # cluster_centers: (n_clusters, n_features)
        # distances: (batch_size, n_clusters)
        distances = torch.cdist(X, self.cluster_centers, p=2) ** 2
        
        # Convert distances to similarities (negative distances)
        similarities = -distances / self.temperature
        
        # Softmax to get soft assignments (probabilities)
        soft_assignments = F.softmax(similarities, dim=1)
        
        return soft_assignments
    
    def initialize_centers(self, X: torch.Tensor):
        """
        Initialize cluster centers using k-means++ or random.
        
        Args:
            X: Input features (n_samples, n_features)
        """
        if self.init_method == "kmeans++":
            # K-means++ initialization
            n_samples = X.shape[0]
            centers = torch.zeros(self.n_clusters, self.n_features, device=X.device)
            
            # First center: random sample
            first_idx = torch.randint(0, n_samples, (1,)).item()
            centers[0] = X[first_idx]
            
            # Subsequent centers: farthest from existing centers
            for i in range(1, self.n_clusters):
                distances = torch.cdist(X, centers[:i], p=2) ** 2
                min_distances = distances.min(dim=1)[0]
                # Sample proportional to squared distances
                probs = min_distances / min_distances.sum()
                next_idx = torch.multinomial(probs, 1).item()
                centers[i] = X[next_idx]
            
            self.cluster_centers.data = centers
        else:
            # Random initialization
            self.cluster_centers.data = X[torch.randperm(X.shape[0])[:self.n_clusters]]
    
    def fit(
        self,
        X: torch.Tensor,
        n_iters: int = 100,
        lr: float = 0.01,
        verbose: bool = True
    ) -> 'DifferentiableKMeans':
        """
        Fit differentiable K-means to data.
        
        Args:
            X: Input features (n_samples, n_features)
            n_iters: Number of optimization iterations
            lr: Learning rate
            verbose: Print progress
        
        Returns:
            Self for chaining
        """
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        
        # Initialize centers
        self.initialize_centers(X)
        
        # Optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        # Training loop
        if verbose:
            pbar = tqdm(range(n_iters), desc="Fitting Differentiable K-means")
        else:
            pbar = range(n_iters)
        
        for iteration in pbar:
            optimizer.zero_grad()
            
            # Get soft assignments
            soft_assignments = self.forward(X)  # (n_samples, n_clusters)
            
            # Compute weighted cluster centers
            # soft_assignments: (n_samples, n_clusters)
            # X: (n_samples, n_features)
            # weighted_centers: (n_clusters, n_features)
            weights = soft_assignments.sum(dim=0, keepdim=True).T  # (n_clusters, 1)
            weighted_centers = (soft_assignments.T @ X) / (weights + 1e-8)  # (n_clusters, n_features)
            
            # Loss: distance between current centers and weighted centers
            loss = F.mse_loss(self.cluster_centers, weighted_centers)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if verbose and iteration % 10 == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        if verbose:
            print(f"[SUCCESS] Differentiable K-means fitted (final loss: {loss.item():.4f})")
        
        return self
    
    def predict_proba(self, X: torch.Tensor) -> np.ndarray:
        """
        Get soft cluster assignment probabilities.
        
        Args:
            X: Input features (n_samples, n_features)
        
        Returns:
            Soft assignments as numpy array (n_samples, n_clusters)
        """
        self.eval()
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X = torch.from_numpy(X).float()
            soft_assignments = self.forward(X)
            return soft_assignments.numpy()
    
    def predict(self, X: torch.Tensor) -> np.ndarray:
        """
        Get hard cluster assignments (argmax of soft assignments).
        
        Args:
            X: Input features (n_samples, n_features)
        
        Returns:
            Hard assignments (n_samples,)
        """
        soft_assignments = self.predict_proba(X)
        return np.argmax(soft_assignments, axis=1)


def main():
    """Test Differentiable K-means on reduced activations."""
    import numpy as np
    
    print("=" * 60)
    print("Shadow - Differentiable K-means Test")
    print("=" * 60)
    
    # Load reduced activations
    try:
        reduced_activations = np.load("reduced_activations.npy")
        print(f"[INFO] Loaded reduced activations: {reduced_activations.shape}")
    except FileNotFoundError:
        print("[ERROR] reduced_activations.npy not found. Run sparse_pca.py first.")
        return
    
    # Convert to torch
    X = torch.from_numpy(reduced_activations).float()
    
    # Initialize and fit
    n_clusters = 10  # Adjust based on your data
    kmeans = DifferentiableKMeans(
        n_clusters=n_clusters,
        n_features=reduced_activations.shape[1],
        temperature=1.0
    )
    
    print(f"[INFO] Fitting {n_clusters} clusters...")
    kmeans.fit(X, n_iters=100, lr=0.01)
    
    # Get soft assignments
    soft_assignments = kmeans.predict_proba(X)
    hard_assignments = kmeans.predict(X)
    
    print(f"[INFO] Soft assignments shape: {soft_assignments.shape}")
    print(f"[INFO] Hard assignments: {np.bincount(hard_assignments)}")
    
    # Save results
    np.save("soft_assignments.npy", soft_assignments)
    np.save("hard_assignments.npy", hard_assignments)
    torch.save(kmeans.state_dict(), "kmeans_model.pt")
    
    print(f"\n[SUCCESS] Differentiable K-means complete!")
    print(f"[INFO] Saved: soft_assignments.npy, hard_assignments.npy, kmeans_model.pt")


if __name__ == "__main__":
    main()

