import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicExpert(nn.Module):
    def __init__(self, feature_in, feature_out):
        super().__init__()
        self.fc = nn.Linear(feature_in, feature_out)

    def forward(self, x):
        return self.fc(x)

class BasicMOE(nn.Module):
    def __init__(self, feature_in, feature_out, num_experts):
        super().__init__()
        self.gate = nn.Linear(feature_in, num_experts)
        # output shape (batch_size, num_experts)
        self.experts = nn.ModuleList(
            BasicExpert(
                feature_in, feature_out
            ) for _ in range(num_experts)
        )

    def forward(self, x):
        # x shape (batch, feature_in)
        # feature_in also calls hidden_size/hidden_dim
        expert_weights = self.gate(x)
        expert_out_list = [
            expert(x) for expert in self.experts
        ] # for each expert, output (batch, feature_in)

        expert_outs = [
            expert_out.unsqueeze(1)
            for expert_out in expert_out_list
        ]

        # expert_out (batch, 1, feature_out)
        expert_out = torch.concat(
            expert_outs,
            dim = 1
        )
        # expert_output shape (b, num_experts, feature_out)

        # expert_weights
        expert_weights = F.softmax(expert_weights, dim=1)
        # (batch, num_experts)

        expert_weights = expert_weights.unsqueeze(1)
        # (batch, 1, num_experts)
        output = expert_weights @ expert_out # (batch, feature_out)
        
        return output.squeeze(1)
    
class MOEConfig:
    def __init__(self, hidden_dim, expert_number, top_k, shared_experts_number=2):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.expert_number = expert_number
        self.top_k = top_k
        self.shared_experts_number = shared_experts_number

class MOERouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate = nn.Linear(config.hidden_dim, config.expert_number)

        # select top k experts
        self.expert_number = config.expert_number
        self.top_k = config.top_k
    
    def forward(self, x):
        # suppose expert_num = 8, top_k = 2
        router_logits = self.gate(x) # (batch * seq_len, expert_number)

        # calculate prob of every expert
        router_probs = F.softmax(router_logits, dim=1, dtype=torch.float)

        # calculate output of top_k experts
        # top_k could do back propagation
        router_weights, selected_experts_indices = torch.topk(
            router_probs,
            self.top_k,
            dim=-1
        ) 
        # router_weights, selected_experts_indices (batch * seq_len, top_k)

        router_weights = router_weights / router_weights.sum(
            dim=-1, keepdim=True
        )

        router_weights = router_weights.to(x.dtype)

        expert_mask = F.one_hot(
            selected_experts_indices,
            num_classes=self.expert_number
        ) 
        # (batch * seq_len, top_k, expert_number)

        expert_mask = expert_mask.permute(2, 1, 0)
        # (expert_number, top_k, batch * seq_len)

        return router_logits, router_weights, selected_experts_indices, expert_mask

class SparseMOE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.top_k = config.top_k
        self.hidden_dim = config.hidden_dim
        self.expert_number = config.expert_number
        
        # initialize expert
        self.experts = nn.ModuleList(
            BasicExpert(
                config.hidden_dim,
                config.hidden_dim
            ) for _ in range(config.expert_number)
        )
        self.router = MOERouter(config)
    
    def forward(self, x):
        # x (batch, seq_len, hidden_dim)
        batch_size, seq_len, hidden_dim = x.size()

        # token dimension calculation, x reshape (batch * seq_len, hidden_len)
        hidden_states = x.view(-1, hidden_dim)

        # do calculation for relevant experts
        router_logits, router_weights, selected_experts_indices, expert_mask = self.router(hidden_states)

        # expert_mask (expert_number, top_k, batch * seq_len)
        # final should be (batch * seq_len, hidden_dim)
        final_hidden_states = torch.zeros(
            (batch_size * seq_len, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device
        )

        # iterate every expert
        # add chosen expert's hidden_states to final_hidden_states
        # # token = batch * seq_len
        for expert_idx in range(self.expert_number):
            expert_layer = self.experts[expert_idx]

            # expert_masks (expert_number, top_k, batch * seq_len)
            current_expert_mask = expert_mask[expert_idx]
            # current_expert_mask (top_k, batch * seq_len)

            router_weights_idx, top_x = torch.where(current_expert_mask)
            # router_weights_idx = 1 or 0
            # top_x is the index of token in batch * seq_len

            # hidden_states (1, batch * seq_len, hidden_dim)
            current_state = hidden_states.unsqueeze(0)[:, top_x, :].reshape(-1, hidden_dim)
            current_state = expert_layer(current_state)
            # current_state (selected_token_number, hidden_dim)
            
            # router_weights (batch * seq_len, top_k)
            current_token_router_weight = router_weights[top_x, router_weights_idx]
            # -> (selected_token_number)

            current_token_router_weight = current_token_router_weight.unsqueeze(-1)
            # -> (selected_token_number, 1)

            current_hidden_states = current_state * current_token_router_weight
            # (selected_token_number, hidden_dim)

            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        final_hidden_states = final_hidden_states.reshape(batch_size, seq_len, hidden_dim)

        return final_hidden_states, router_logits # (batch_size * seq_len, expert_number)

class SharedExpertMOE(nn.Module):
    def __init__(self, config):
        super().__init__()
    
        self.config = config
        self.routed_experts_moe = SparseMOE(config)
        self.shared_experts = nn.ModuleList(
            BasicExpert(
                self.config.hidden_dim,
                self.config.hidden_dim
            ) for _ in range(config.shared_experts_number)
        )

    def forward(self, x):
        # x (batch_size, seq_len, hidden_dim)
        batch_size, seq_len, hidden_dim = x.size()

        shared_experts_output_list = [
            expert(x) for expert in self.shared_experts
        ]
        shared_expert_output = torch.stack(
            shared_experts_output_list, 
            dim=0
        )
        # (shared_experts_number, batch_size, seq_len, hidden_dim)

        shared_expert_output = shared_expert_output.sum(dim=0, keepdim=False)
        # (batch_size, seq_len, hidden_dim)
        sparse_moe_out, router_logits = self.routed_experts_moe(x)

        output = shared_expert_output + sparse_moe_out
        return output, router_logits

def test_basic_moe():
    x = torch.rand(4, 512)
    basic_moe = BasicMOE(512, 128, 4)
    output = basic_moe(x)
    print(f"Basic MOE: {output.shape}")

test_basic_moe()

print("-------------------------------")

def test_sparse_moe():
    x = torch.rand(2, 4, 16)
    config = MOEConfig(16, 4, 2)
    token_level_moe = SparseMOE(config)
    out = token_level_moe(x)
    print(f"Sparse MOE: {out[0].shape}, {out[1].shape}")

test_sparse_moe()

print("-------------------------------")

def test_share_moe():
    x = torch.rand(2, 4, 16)
    config = MOEConfig(16, 4, 2, shared_experts_number=2)
    share_expert_moe = SharedExpertMOE(config)
    out = share_expert_moe(x)
    print(f"Share expert MOE: {out[0].shape}, {out[1].shape}")

test_share_moe()

print("-------------------------------")
print("-------------------------------")

def switch_load_balancing_loss(router_logits:torch.Tensor, num_experts:int, top_k:int):
    # total loss = auxiliary_loss + z_loss
    router_probs = torch.softmax(router_logits, dim=-1)
    _, selected_experts = torch.topk(router_probs, k=top_k, dim=-1)
    mask = F.one_hot(selected_experts, num_experts).float()
    expect_load = torch.ones_like(router_probs) / num_experts

    actual_load = mask.mean(dim=0)

    aux_loss = torch.sum(actual_load * router_probs.mean(dim=0)) * num_experts

    z_loss = torch.mean(torch.square(router_logits))
    z_loss_weight = 0.001

    total_loss = aux_loss + z_loss * z_loss_weight

    return total_loss

def calculate_expert_load(router_logits, num_experts, top_k):
    router_probs = torch.softmax(router_logits, dim=-1)

    _,selected_experts = torch.topk(
        router_probs,
        k=top_k,
        dim=-1
    )

    expert_mask = F.one_hot(
        selected_experts,
        num_classes=num_experts
    ).float()

    expert_load = expert_mask.mean(dim=(0,1))
    return expert_load

def test_moe_training():
    batch_size = 32
    seq_len = 16
    hidden_dim = 32
    num_batches = 100

    config = MOEConfig(
        hidden_dim=hidden_dim,
        expert_number=4,
        top_k=2,
        shared_experts_number=2
    )

    model = SharedExpertMOE(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for batch in range(num_batches):
        x = torch.randn(batch_size, seq_len, hidden_dim)
        target = torch.randn(batch_size, seq_len, hidden_dim)

        output, router_logits = model(x)

        mse_loss = F.mse_loss(output, target)

        aux_loss = switch_load_balancing_loss(router_logits, config.expert_number, config.top_k)

        total_loss = mse_loss + 0.01 * aux_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if batch % 10 == 0 :
            expert_load = calculate_expert_load(
                router_logits,
                config.expert_number,
                config.top_k
            )
            print(f"Expert load: {expert_load.tolist()}\n"
                  f"Batch {batch}, Loss {total_loss.item():.4f}"
                  f" || MSE: {mse_loss.item():.4f}, Aux: {aux_loss.item():.4f}\n")
            
test_moe_training()
