import fire

from rover.data_utils.embed import embed


def te2hf(input_path, output_path):
    from rover.models.te_llama.te_llama import te_to_hf

    te_to_hf(input_path, output_path)


if __name__ == "__main__":
    fire.Fire(
        dict(
            te_to_hf=te2hf,
            embed=embed,
        )
    )
