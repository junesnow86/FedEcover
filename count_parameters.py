from transformers import BertModel

from modules.models import GPT2DecoderBlock, Transformer


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_ffn_parameters(model, type="transformer"):
    ffn_params = 0
    if type == "bert":
        for layer in model.encoder.layer:
            ffn_params += count_parameters(layer.intermediate)
            ffn_params += count_parameters(layer.output.dense)
    elif type == "transformer":
        for encoder in model.encoder_blocks:
            ffn_params += count_parameters(encoder.ffn)
        for decoder in model.decoder_blocks:
            ffn_params += count_parameters(decoder.ffn)
    elif type == "gpt2decoder":
        ffn_params += count_parameters(model.ffn)
    return ffn_params


def count_transformer_parameters():
    # 定义模型参数
    src_vocab_size = 30522
    tgt_vocab_size = 30522
    num_heads = 8
    num_layers = 6
    d_model = 512
    d_ff = 2048
    max_seq_len = 512

    # 初始化模型
    model = Transformer(
        src_vocab_size,
        tgt_vocab_size,
        num_heads,
        num_layers,
        d_model,
        d_ff,
        max_seq_len,
    )

    # 计算总参数量和 FFN 参数量
    total_params = count_parameters(model)
    ffn_params = count_ffn_parameters(model)

    # 计算 FFN 参数量占比
    ffn_ratio = ffn_params / total_params * 100
    print(f"Total parameters: {total_params}")
    print(f"FFN parameters: {ffn_params}")
    print(f"FFN parameter ratio: {ffn_ratio:.2f}%")


def count_bert_parameters():
    model = BertModel.from_pretrained("bert-base-uncased")
    total_params = count_parameters(model)
    ffn_params = count_ffn_parameters(model)

    ffn_ratio = ffn_params / total_params * 100
    print(f"Total parameters: {total_params}")
    print(f"FFN parameters: {ffn_params}")
    print(f"FFN parameter ratio: {ffn_ratio:.2f}%")


def count_gpt2_decoder_parameters():
    model = GPT2DecoderBlock()
    total_params = count_parameters(model)
    ffn_params = count_ffn_parameters(model, type="gpt2decoder")

    ffn_ratio = ffn_params / total_params * 100
    print(f"Total parameters: {total_params}")
    print(f"FFN parameters: {ffn_params}")
    print(f"FFN parameter ratio: {ffn_ratio:.2f}%")


if __name__ == "__main__":
    # count_transformer_parameters()
    count_gpt2_decoder_parameters()
