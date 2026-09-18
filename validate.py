import torch
from prompt import EPrompt

# --- Test 1: baseline (dilation=1) ---
ep1 = EPrompt(
    length=20, embed_dim=768, num_tasks=10, kernel_size=3,
    prompt_pool=True, prompt_key=True,
    num_layers=1, use_prefix_tune_for_e_prompt=True,
    num_heads=12, prompts_per_task=2, dilation_rate=1
)
assert ep1.dilation == 1, 'FAIL: dilation should be 1'
print('Test 1 PASS — dilation=1, self.dilation=', ep1.dilation)

# --- Test 2: dilated (dilation=2) ---
ep2 = EPrompt(
    length=20, embed_dim=768, num_tasks=10, kernel_size=3,
    prompt_pool=True, prompt_key=True,
    num_layers=1, use_prefix_tune_for_e_prompt=True,
    num_heads=12, prompts_per_task=2, dilation_rate=2
)
assert ep2.dilation == 2, 'FAIL: dilation should be 2'
print('Test 2 PASS — dilation=2, self.dilation=', ep2.dilation)

# --- Test 3: forward pass with dilation=2 ---
ep2.process_new_task(0, 2)
x_embed = torch.randn(4, 197, 768)
cls_feat = torch.randn(4, 768)
out = ep2(x_embed, task_id=0, layer_num=0, cls_features=cls_feat)
print('Test 3 PASS — batched_prompt shape:', out['batched_prompt'].shape)

print('ALL TESTS PASSED.')
