import torch
import torch.nn as nn


def svd_flip(u, v):
    # columns of u, rows of v
    max_abs_cols = torch.argmax(torch.abs(u), 0)
    i = torch.arange(u.shape[1]).to(u.device)
    signs = torch.sign(u[max_abs_cols, i])
    u *= signs
    v *= signs.view(-1, 1)
    return u, v


class PCA(nn.Module):
    def __init__(self, n_components):
        super().__init__()
        self.n_components = n_components

    @torch.no_grad()
    def fit(self, X):
        n, d = X.size()
        if self.n_components is not None:
            d = min(self.n_components, d)
        self.register_buffer("mean_", X.mean(0, keepdim=True))
        Z = X - self.mean_ # center
        U, S, Vh = torch.linalg.svd(Z, full_matrices=False)
        Vt = Vh
        U, Vt = svd_flip(U, Vt)
        self.register_buffer("components_", Vt[:d])
        return self

    def forward(self, X):
        return self.transform(X)

    def transform(self, X):
        assert hasattr(self, "components_"), "PCA must be fit before use."
        return torch.matmul(X - self.mean_, self.components_.t())

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, Y):
        assert hasattr(self, "components_"), "PCA must be fit before use."
        return torch.matmul(Y, self.components_) + self.mean_


def compute_icv(model, xs, ys, entire_prompt=True):
    if entire_prompt:
        x_act_idx = -2
        y_act_idx = -1
    else:
        x_act_idx = 0
        y_act_idx = 1

    with torch.no_grad():
        _, out = model.get_transformer_out(xs, ys)

    H_x, H_y = [], []
    for hidden_state in out.hidden_states:
        H_x.append(hidden_state[:, x_act_idx, :])
        H_y.append(hidden_state[:, y_act_idx, :])

    num_layers, n_dims = len(H_x), H_x[0].shape[-1]

    H_x_flatten = torch.cat(H_x, dim=1)
    H_y_flatten = torch.cat(H_y, dim=1)

    diff_H = H_y_flatten - H_x_flatten
    H = diff_H / torch.norm(diff_H, p=2, dim=1).reshape(-1, 1)

    pca = PCA(n_components=1).to(H.device).fit(H.float())
    icv = (pca.components_.sum(dim=0, keepdim=True) + pca.mean_).mean(0)
    icv = icv.view(num_layers, n_dims)

    return icv


def add_icv(model, xs, ys, icv, lambda_=0.0):
    """
    Add custom vectors to the output of each layer in GPT-2 and return the final modified output.

    :param model: GPT-2 model instance.
    :param input_ids: Tensor of input ids.
    :param custom_vectors: Tensor of shape (L, d) where L is the number of layers and d is the embedding dimension.
    :return: The final output of the model after modifications.
    """

    # Function to be applied at each layer
    hooks = []

    def create_hook_function(layer_idx):
        # Define the hook function
        def hook_function(module, input, output):
            # Modify the first element of the output tuple (which are the hidden states)
            modified_output = output[0] + lambda_ * icv[layer_idx]
            scale = torch.norm(output[0], p=2, dim=-1) / torch.norm(modified_output, p=2, dim=-1)
            scale = scale.unsqueeze(-1)
            modified_output *= scale

            # Return a new tuple with the modified output and other elements unchanged
            return (modified_output,) + output[1:]

        return hook_function

    # Register the hook to each layer
    hooks = []
    for i, layer in enumerate(model._backbone.h):
        hook = layer.register_forward_hook(create_hook_function(i))
        hooks.append(hook)

    # Pass the input data through the model
    with torch.no_grad():
        _, model_out = model.get_transformer_out(xs, ys)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return model_out  # This is the modified output