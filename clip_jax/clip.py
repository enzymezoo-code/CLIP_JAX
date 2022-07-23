
def convert_params(torch_state, jax_params):
    def name_iter(pytree, root, f):
        new_out = {}
        for k, v in pytree.items():
            if isinstance(v, dict):
                new_out[k] = name_iter(v, root + "/" + k, f)
            else:
                new_out[k] = f(v, root + "/" + k)
        return new_out

    def process_node(value, name):
        name = name.lstrip("/")
        tensor_name = name.split("/")[-1]
        tensor_name = {
            "w": "weight",
            "b": "bias",
            "scale": "weight",
            "offset": "bias",
            "embeddings": "weight",
        }.get(tensor_name, tensor_name)

        tensor_path = "/".join(name.split("/")[:-1]).replace("/~/", ".").replace("/", ".").replace("resblocks",
                                                                                                   "resblocks.").replace(
            "~", "")
        new_tensor = value

        pytorch_name = tensor_path + "." + tensor_name if tensor_path else tensor_name

        if "conv" in name:
            pytorch_path = tensor_path + "." + tensor_name
            pytorch_tensor = torch_state[pytorch_path].permute([2, 3, 1, 0])
            new_tensor = jnp.array(pytorch_tensor)
        elif pytorch_name in torch_state:
            pytorch_tensor = torch_state[pytorch_name]

            if tensor_name == "weight" and "token_embedding" not in tensor_path:
                pytorch_tensor = pytorch_tensor.t()

            new_tensor = jnp.array(pytorch_tensor)
        else:
            raise Exception("not implemented")

        assert new_tensor.shape == value.shape
        return new_tensor.astype("float32")

    return name_iter(jax_params, "", process_node)