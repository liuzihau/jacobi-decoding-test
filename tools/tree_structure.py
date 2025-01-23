from copy import copy
import torch

class Node:
    def __init__(self, val, prob, parent=None, children=None):
        self.val = val
        self.prob = prob
        self.parent = parent
        self.children = children if children is not None else []
        self.level = self.parent.level + 1 if self.parent is not None else 0

        self.route = None
        self.route_prob = self.parent.route_prob * prob if self.parent is not None else self.prob
            
    def add_child(self, node):
        self.children.append(node)
    
    def update_route(self, index):
        self.route = [index] if self.parent is None else self.parent.route + [index]

    def __repr__(self):
        return f"(Node val:{self.val}, parent:{self.parent}, children:{self.children})"

class TreeStructure:
    def __init__(self, root_val):
        self.index_dict = {}
        self.layers = []
        self.root = self.add_node(root_val, 1, None, [])
    
    def add_node(self, val, prob, parent, children):
        node = Node(val, prob, parent, children)
        if parent is not None:
            parent.add_child(node)
        self.update_layers(node)
        return node

    def update_layers(self, node):
        level_idx = node.level
        while len(self.layers) <= level_idx:
            self.layers.append([])
        self.layers[level_idx].append(node)

    def reset_index_dict(self):
        self.index_dict = {}

    def __len__(self):
        return sum(len(layer) for layer in self.layers)
    
    def __repr__(self):
        s = ""
        for i, layer in enumerate(self.layers):
            s += f"[{i}th layer] ["
            for node in layer:
                parent = node.parent.val if node.parent is not None else None
                s += f"(val:{node.val}, parent:{parent}), "
            s = s[:-2] + "]\n"
        return s

class InputProcessor:
    def __init__(self, input_dtype, attention_dtype, loss_mask_dtype, device, jacobi_token_nums, jacobi_id=0):
        self.jacobi_token_nums = jacobi_token_nums
        self.jacobi_id = jacobi_id
        self.input_dtype = input_dtype
        self.attention_dtype = attention_dtype
        self.loss_mask_dtype = loss_mask_dtype
        self.min_dtype = torch.finfo(self.attention_dtype).min
        self.device = device

    def build_inputs(self, tree):
        length = len(tree) * (1+self.jacobi_token_nums)
        deny_mask = torch.full((length, length), fill_value=self.min_dtype, dtype=self.attention_dtype, device=self.device)
        attention_mask = torch.ones((length, length), dtype=self.attention_dtype, device=self.device)
        diagonal_mask = torch.arange(1+self.jacobi_token_nums, dtype=self.attention_dtype, device=self.device)
        diagonal_mask = diagonal_mask > diagonal_mask.reshape(-1, 1)
        tree.reset_index_dict()
        input_ids = []
        loss_mask = []
        index = 0
        for layer in tree.layers:
            for node in layer:
                tree.index_dict[node] = index
                node.update_route(index)

                if node.parent is not None:
                    current_mask = attention_mask[tree.index_dict[node.parent]]
                else:
                    current_mask = attention_mask[0]
                input_ids += [node.val] + [self.jacobi_id] * self.jacobi_token_nums
                loss_mask += [0] + [1] * self.jacobi_token_nums
                attention_mask[index:index+1+self.jacobi_token_nums, :] = current_mask
                attention_mask[index:index+1+self.jacobi_token_nums, index:index+1+self.jacobi_token_nums] = diagonal_mask
                index += 1+self.jacobi_token_nums
        
        cache_position = ((attention_mask - 1)* -1).type(torch.int32).sum(dim=-1) - 1
        attention_mask = (attention_mask * deny_mask).unsqueeze(0)
        input_ids = torch.tensor(input_ids, dtype=self.input_dtype, device=self.device)
        loss_mask = torch.tensor(loss_mask, dtype=self.loss_mask_dtype, device=self.device)
        return input_ids, attention_mask, loss_mask, cache_position


def test_tree_structure():
    # Initialize Tree
    tree = TreeStructure(root_val=0)
    assert len(tree) == 1
    assert tree.root.val == 0
    assert tree.root.prob == 1
    assert tree.root.level == 0
    assert tree.root.route_prob == 1

    # Add First Layer Nodes
    node1 = tree.add_node(val=1, prob=0.6, parent=tree.root, children=[])
    node2 = tree.add_node(val=2, prob=0.4, parent=tree.root, children=[])
    assert len(tree) == 3  # root + 2 children
    assert len(tree.layers) == 2  # root (level 0), children (level 1)
    assert tree.root.children == [node1, node2]

    # Check Parent-Child Relationship
    assert node1.parent == tree.root
    assert node2.parent == tree.root
    assert node1.level == 1
    assert node2.level == 1
    assert node1.route_prob == 0.6
    assert node2.route_prob == 0.4

    # Add Second Layer Nodes
    node3 = tree.add_node(val=3, prob=0.7, parent=node1, children=[])
    node4 = tree.add_node(val=4, prob=0.3, parent=node1, children=[])
    assert len(tree) == 5  # root + 4 children
    assert len(tree.layers) == 3  # root (level 0), layer 1, layer 2
    assert node3.level == 2
    assert node4.level == 2
    assert node3.route_prob == 0.42  # 0.6 * 0.7
    assert node4.route_prob == 0.18  # 0.6 * 0.3

    # Validate Dynamic Layer Expansion
    node5 = tree.add_node(val=5, prob=1.0, parent=node4, children=[])
    assert len(tree) == 6
    assert len(tree.layers) == 4  # New layer added dynamically
    assert node5.route_prob == 0.18  # 0.6 * 0.3 * 1.0

    # Validate Mutable Children Fix
    assert node1.children == [node3, node4]
    assert node5.children == []



    # Validate Dynamic Layer Expansion
    node6 = tree.add_node(val=6, prob=1.0, parent=node2, children=[])
    assert len(tree) == 7
    assert len(tree.layers) == 4  # New layer added dynamically
    assert node6.route_prob == 0.4  # 0.6 * 0.3 * 1.0

    # Validate Mutable Children Fix
    assert node2.children == [node6]
    assert node6.children == []

    print("All tests passed!")
    return tree

if __name__ == "__main__":
    a = Node(1, 1)
    index_dict = {}
    index_dict[a] =3
    print(index_dict[a])

    b = [0, 1, 2]
    c = copy(b)
    c[1]= 3
    print(b, c)

    diagonal_mask = torch.arange(1+2, dtype=torch.int32)
    diagonal_mask = diagonal_mask > diagonal_mask.reshape(-1, 1)
    print(diagonal_mask)
    attention_mask = torch.ones((4, 4), dtype=torch.bool)
    print(attention_mask)

    tree = test_tree_structure()

    input_processor = InputProcessor(
        input_dtype=torch.long, attention_dtype=torch.float32, loss_mask_dtype=torch.long, device="cuda", jacobi_token_nums=2, jacobi_id=87)
    input_ids, final_mask, loss_mask = input_processor.build_inputs(tree)
    print(input_ids)
    print(loss_mask)
    print("sttention mask:")
    final_mask /= final_mask.min()
    for mask in final_mask:
        print(mask.type(torch.int16).detach().cpu().tolist())