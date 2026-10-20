// Generated from locked comfy/utils.py; source SHA256: 9320cf5c624e96acc1d9ec92ff120ed6387a2142707f3194cb21dd0d11ac2500
// Literal source order is retained. Hash state and CPython construction run natively.
use super::detector_values::*;
use crate::{native::Error, patching::options::Options};
pub(super) const UNET_MAP_ATTENTIONS: &[&str] = &[
    "proj_in.weight",
    "proj_in.bias",
    "proj_out.weight",
    "proj_out.bias",
    "norm.weight",
    "norm.bias",
];
pub(super) const UNET_MAP_ATTENTIONS_MARSHALLED: &[&str] = &[
    "norm.bias",
    "norm.weight",
    "proj_in.bias",
    "proj_out.bias",
    "proj_in.weight",
    "proj_out.weight",
];
pub(super) const TRANSFORMER_BLOCKS: &[&str] = &[
    "norm1.weight",
    "norm1.bias",
    "norm2.weight",
    "norm2.bias",
    "norm3.weight",
    "norm3.bias",
    "attn1.to_q.weight",
    "attn1.to_k.weight",
    "attn1.to_v.weight",
    "attn1.to_out.0.weight",
    "attn1.to_out.0.bias",
    "attn2.to_q.weight",
    "attn2.to_k.weight",
    "attn2.to_v.weight",
    "attn2.to_out.0.weight",
    "attn2.to_out.0.bias",
    "ff.net.0.proj.weight",
    "ff.net.0.proj.bias",
    "ff.net.2.weight",
    "ff.net.2.bias",
];
pub(super) const TRANSFORMER_BLOCKS_MARSHALLED: &[&str] = &[
    "norm1.bias",
    "norm2.bias",
    "norm3.bias",
    "norm1.weight",
    "norm2.weight",
    "norm3.weight",
    "ff.net.2.bias",
    "ff.net.2.weight",
    "attn1.to_k.weight",
    "attn1.to_q.weight",
    "attn1.to_v.weight",
    "attn2.to_k.weight",
    "attn2.to_q.weight",
    "attn2.to_v.weight",
    "ff.net.0.proj.bias",
    "attn1.to_out.0.bias",
    "attn2.to_out.0.bias",
    "ff.net.0.proj.weight",
    "attn1.to_out.0.weight",
    "attn2.to_out.0.weight",
];
pub(super) const UNET_MAP_RESNET: &[(&str, &str)] = &[
    ("in_layers.2.weight", "conv1.weight"),
    ("in_layers.2.bias", "conv1.bias"),
    ("emb_layers.1.weight", "time_emb_proj.weight"),
    ("emb_layers.1.bias", "time_emb_proj.bias"),
    ("out_layers.3.weight", "conv2.weight"),
    ("out_layers.3.bias", "conv2.bias"),
    ("skip_connection.weight", "conv_shortcut.weight"),
    ("skip_connection.bias", "conv_shortcut.bias"),
    ("in_layers.0.weight", "norm1.weight"),
    ("in_layers.0.bias", "norm1.bias"),
    ("out_layers.0.weight", "norm2.weight"),
    ("out_layers.0.bias", "norm2.bias"),
];
pub(super) const UNET_MAP_BASIC: &[(&str, &str)] = &[
    ("label_emb.0.0.weight", "class_embedding.linear_1.weight"),
    ("label_emb.0.0.bias", "class_embedding.linear_1.bias"),
    ("label_emb.0.2.weight", "class_embedding.linear_2.weight"),
    ("label_emb.0.2.bias", "class_embedding.linear_2.bias"),
    ("label_emb.0.0.weight", "add_embedding.linear_1.weight"),
    ("label_emb.0.0.bias", "add_embedding.linear_1.bias"),
    ("label_emb.0.2.weight", "add_embedding.linear_2.weight"),
    ("label_emb.0.2.bias", "add_embedding.linear_2.bias"),
    ("input_blocks.0.0.weight", "conv_in.weight"),
    ("input_blocks.0.0.bias", "conv_in.bias"),
    ("out.0.weight", "conv_norm_out.weight"),
    ("out.0.bias", "conv_norm_out.bias"),
    ("out.2.weight", "conv_out.weight"),
    ("out.2.bias", "conv_out.bias"),
    ("time_embed.0.weight", "time_embedding.linear_1.weight"),
    ("time_embed.0.bias", "time_embedding.linear_1.bias"),
    ("time_embed.2.weight", "time_embedding.linear_2.weight"),
    ("time_embed.2.bias", "time_embedding.linear_2.bias"),
];
pub(super) const UNET_MAP_BASIC_MARSHALLED: &[(&str, &str)] = &[
    ("out.0.bias", "conv_norm_out.bias"),
    ("out.2.bias", "conv_out.bias"),
    ("out.0.weight", "conv_norm_out.weight"),
    ("out.2.weight", "conv_out.weight"),
    ("time_embed.0.bias", "time_embedding.linear_1.bias"),
    ("time_embed.2.bias", "time_embedding.linear_2.bias"),
    ("label_emb.0.0.bias", "add_embedding.linear_1.bias"),
    ("label_emb.0.0.bias", "class_embedding.linear_1.bias"),
    ("label_emb.0.2.bias", "add_embedding.linear_2.bias"),
    ("label_emb.0.2.bias", "class_embedding.linear_2.bias"),
    ("time_embed.0.weight", "time_embedding.linear_1.weight"),
    ("time_embed.2.weight", "time_embedding.linear_2.weight"),
    ("label_emb.0.0.weight", "add_embedding.linear_1.weight"),
    ("label_emb.0.0.weight", "class_embedding.linear_1.weight"),
    ("label_emb.0.2.weight", "add_embedding.linear_2.weight"),
    ("label_emb.0.2.weight", "class_embedding.linear_2.weight"),
    ("input_blocks.0.0.bias", "conv_in.bias"),
    ("input_blocks.0.0.weight", "conv_in.weight"),
];
pub(super) fn run(mut cx: Context) -> Result<Options, Error> {
    // comfy/utils.py:375
    if cx.truth(&{
        let detector_tmp_1 = text("num_res_blocks");
        let detector_tmp_2 = cx.get("unet_config")?;
        cx.compare("NotIn", &detector_tmp_1, &detector_tmp_2)?
    })? {
        // comfy/utils.py:376
        return cx.mapping(vec![]);
    }
    // comfy/utils.py:377
    cx.set(
        "num_res_blocks",
        cx.index(&cx.get("unet_config")?, &text("num_res_blocks"))?,
    );
    // comfy/utils.py:378
    cx.set(
        "channel_mult",
        cx.index(&cx.get("unet_config")?, &text("channel_mult"))?,
    );
    // comfy/utils.py:379
    cx.set(
        "transformer_depth",
        cx.slice(
            &cx.index(&cx.get("unet_config")?, &text("transformer_depth"))?,
            None,
            None,
            None,
        )?,
    );
    // comfy/utils.py:380
    cx.set(
        "transformer_depth_output",
        cx.slice(
            &cx.index(&cx.get("unet_config")?, &text("transformer_depth_output"))?,
            None,
            None,
            None,
        )?,
    );
    // comfy/utils.py:381
    cx.set(
        "num_blocks",
        cx.function("len", vec![cx.get("channel_mult")?], vec![])?,
    );
    // comfy/utils.py:383
    cx.set(
        "transformers_mid",
        cx.method(
            &cx.get("unet_config")?,
            "get",
            vec![text("transformer_depth_middle"), none()],
            vec![],
        )?,
    );
    // comfy/utils.py:385
    cx.set("diffusers_unet_map", cx.mapping(vec![])?);
    // comfy/utils.py:386
    for detector_item in cx.iter(&cx.function("range", vec![cx.get("num_blocks")?], vec![])?)? {
        cx.set("x", detector_item); // comfy/utils.py:387
        cx.set(
            "n",
            cx.binary(
                "Add",
                &integer(1_i128)?,
                &cx.binary(
                    "Mult",
                    &cx.binary(
                        "Add",
                        &cx.index(&cx.get("num_res_blocks")?, &cx.get("x")?)?,
                        &integer(1_i128)?,
                    )?,
                    &cx.get("x")?,
                )?,
            )?,
        );
        // comfy/utils.py:388
        for detector_item in cx.iter(&cx.function(
            "range",
            vec![cx.index(&cx.get("num_res_blocks")?, &cx.get("x")?)?],
            vec![],
        )?)? {
            cx.set("i", detector_item); // comfy/utils.py:389
            for detector_item in cx.iter(&cx.get("UNET_MAP_RESNET")?)? {
                cx.set("b", detector_item); // comfy/utils.py:390
                cx.set_item(
                    &cx.get("diffusers_unet_map")?,
                    &cx.method(
                        &text("down_blocks.{}.resnets.{}.{}"),
                        "format",
                        vec![
                            cx.get("x")?,
                            cx.get("i")?,
                            cx.index(&cx.get("UNET_MAP_RESNET")?, &cx.get("b")?)?,
                        ],
                        vec![],
                    )?,
                    cx.method(
                        &text("input_blocks.{}.0.{}"),
                        "format",
                        vec![cx.get("n")?, cx.get("b")?],
                        vec![],
                    )?,
                )?;
            }
            // comfy/utils.py:391
            cx.set(
                "num_transformers",
                cx.method(
                    &cx.get("transformer_depth")?,
                    "pop",
                    vec![integer(0_i128)?],
                    vec![],
                )?,
            );
            // comfy/utils.py:392
            if cx.truth(&{
                let detector_tmp_3 = cx.get("num_transformers")?;
                let detector_tmp_4 = integer(0_i128)?;
                cx.compare("Gt", &detector_tmp_3, &detector_tmp_4)?
            })? {
                // comfy/utils.py:393
                for detector_item in cx.iter(&cx.get("UNET_MAP_ATTENTIONS")?)? {
                    cx.set("b", detector_item); // comfy/utils.py:394
                    cx.set_item(
                        &cx.get("diffusers_unet_map")?,
                        &cx.method(
                            &text("down_blocks.{}.attentions.{}.{}"),
                            "format",
                            vec![cx.get("x")?, cx.get("i")?, cx.get("b")?],
                            vec![],
                        )?,
                        cx.method(
                            &text("input_blocks.{}.1.{}"),
                            "format",
                            vec![cx.get("n")?, cx.get("b")?],
                            vec![],
                        )?,
                    )?;
                }
                // comfy/utils.py:395
                for detector_item in
                    cx.iter(&cx.function("range", vec![cx.get("num_transformers")?], vec![])?)?
                {
                    cx.set("t", detector_item); // comfy/utils.py:396
                    for detector_item in cx.iter(&cx.get("TRANSFORMER_BLOCKS")?)? {
                        cx.set("b", detector_item); // comfy/utils.py:397
                        cx.set_item(
                            &cx.get("diffusers_unet_map")?,
                            &cx.method(
                                &text("down_blocks.{}.attentions.{}.transformer_blocks.{}.{}"),
                                "format",
                                vec![cx.get("x")?, cx.get("i")?, cx.get("t")?, cx.get("b")?],
                                vec![],
                            )?,
                            cx.method(
                                &text("input_blocks.{}.1.transformer_blocks.{}.{}"),
                                "format",
                                vec![cx.get("n")?, cx.get("t")?, cx.get("b")?],
                                vec![],
                            )?,
                        )?;
                    }
                }
            }
            // comfy/utils.py:398
            cx.set("n", cx.binary("Add", &cx.get("n")?, &integer(1_i128)?)?);
        }
        // comfy/utils.py:399
        for detector_item in cx.iter(&list(vec![text("weight"), text("bias")]))? {
            cx.set("k", detector_item); // comfy/utils.py:400
            cx.set_item(
                &cx.get("diffusers_unet_map")?,
                &cx.method(
                    &text("down_blocks.{}.downsamplers.0.conv.{}"),
                    "format",
                    vec![cx.get("x")?, cx.get("k")?],
                    vec![],
                )?,
                cx.method(
                    &text("input_blocks.{}.0.op.{}"),
                    "format",
                    vec![cx.get("n")?, cx.get("k")?],
                    vec![],
                )?,
            )?;
        }
    }
    // comfy/utils.py:402
    cx.set("i", integer(0_i128)?);
    // comfy/utils.py:403
    for detector_item in cx.iter(&cx.get("UNET_MAP_ATTENTIONS")?)? {
        cx.set("b", detector_item); // comfy/utils.py:404
        cx.set_item(
            &cx.get("diffusers_unet_map")?,
            &cx.method(
                &text("mid_block.attentions.{}.{}"),
                "format",
                vec![cx.get("i")?, cx.get("b")?],
                vec![],
            )?,
            cx.method(
                &text("middle_block.1.{}"),
                "format",
                vec![cx.get("b")?],
                vec![],
            )?,
        )?;
    }
    // comfy/utils.py:405
    for detector_item in
        cx.iter(&cx.function("range", vec![cx.get("transformers_mid")?], vec![])?)?
    {
        cx.set("t", detector_item); // comfy/utils.py:406
        for detector_item in cx.iter(&cx.get("TRANSFORMER_BLOCKS")?)? {
            cx.set("b", detector_item); // comfy/utils.py:407
            cx.set_item(
                &cx.get("diffusers_unet_map")?,
                &cx.method(
                    &text("mid_block.attentions.{}.transformer_blocks.{}.{}"),
                    "format",
                    vec![cx.get("i")?, cx.get("t")?, cx.get("b")?],
                    vec![],
                )?,
                cx.method(
                    &text("middle_block.1.transformer_blocks.{}.{}"),
                    "format",
                    vec![cx.get("t")?, cx.get("b")?],
                    vec![],
                )?,
            )?;
        }
    }
    // comfy/utils.py:409
    for detector_item in cx.iter(&cx.function(
        "enumerate",
        vec![list(vec![integer(0_i128)?, integer(2_i128)?])],
        vec![],
    )?)? {
        let detector_tmp_5 = detector_item;
        cx.set("i", cx.index(&detector_tmp_5, &integer(0)?)?);
        cx.set("n", cx.index(&detector_tmp_5, &integer(1)?)?); // comfy/utils.py:410
        for detector_item in cx.iter(&cx.get("UNET_MAP_RESNET")?)? {
            cx.set("b", detector_item); // comfy/utils.py:411
            cx.set_item(
                &cx.get("diffusers_unet_map")?,
                &cx.method(
                    &text("mid_block.resnets.{}.{}"),
                    "format",
                    vec![
                        cx.get("i")?,
                        cx.index(&cx.get("UNET_MAP_RESNET")?, &cx.get("b")?)?,
                    ],
                    vec![],
                )?,
                cx.method(
                    &text("middle_block.{}.{}"),
                    "format",
                    vec![cx.get("n")?, cx.get("b")?],
                    vec![],
                )?,
            )?;
        }
    }
    // comfy/utils.py:413
    cx.set(
        "num_res_blocks",
        cx.function(
            "list",
            vec![cx.function("reversed", vec![cx.get("num_res_blocks")?], vec![])?],
            vec![],
        )?,
    );
    // comfy/utils.py:414
    for detector_item in cx.iter(&cx.function("range", vec![cx.get("num_blocks")?], vec![])?)? {
        cx.set("x", detector_item); // comfy/utils.py:415
        cx.set(
            "n",
            cx.binary(
                "Mult",
                &cx.binary(
                    "Add",
                    &cx.index(&cx.get("num_res_blocks")?, &cx.get("x")?)?,
                    &integer(1_i128)?,
                )?,
                &cx.get("x")?,
            )?,
        );
        // comfy/utils.py:416
        cx.set(
            "l",
            cx.binary(
                "Add",
                &cx.index(&cx.get("num_res_blocks")?, &cx.get("x")?)?,
                &integer(1_i128)?,
            )?,
        );
        // comfy/utils.py:417
        for detector_item in cx.iter(&cx.function("range", vec![cx.get("l")?], vec![])?)? {
            cx.set("i", detector_item); // comfy/utils.py:418
            cx.set("c", integer(0_i128)?);
            // comfy/utils.py:419
            for detector_item in cx.iter(&cx.get("UNET_MAP_RESNET")?)? {
                cx.set("b", detector_item); // comfy/utils.py:420
                cx.set_item(
                    &cx.get("diffusers_unet_map")?,
                    &cx.method(
                        &text("up_blocks.{}.resnets.{}.{}"),
                        "format",
                        vec![
                            cx.get("x")?,
                            cx.get("i")?,
                            cx.index(&cx.get("UNET_MAP_RESNET")?, &cx.get("b")?)?,
                        ],
                        vec![],
                    )?,
                    cx.method(
                        &text("output_blocks.{}.0.{}"),
                        "format",
                        vec![cx.get("n")?, cx.get("b")?],
                        vec![],
                    )?,
                )?;
            }
            // comfy/utils.py:421
            cx.set("c", cx.binary("Add", &cx.get("c")?, &integer(1_i128)?)?);
            // comfy/utils.py:422
            cx.set(
                "num_transformers",
                cx.method(&cx.get("transformer_depth_output")?, "pop", vec![], vec![])?,
            );
            // comfy/utils.py:423
            if cx.truth(&{
                let detector_tmp_6 = cx.get("num_transformers")?;
                let detector_tmp_7 = integer(0_i128)?;
                cx.compare("Gt", &detector_tmp_6, &detector_tmp_7)?
            })? {
                // comfy/utils.py:424
                cx.set("c", cx.binary("Add", &cx.get("c")?, &integer(1_i128)?)?);
                // comfy/utils.py:425
                for detector_item in cx.iter(&cx.get("UNET_MAP_ATTENTIONS")?)? {
                    cx.set("b", detector_item); // comfy/utils.py:426
                    cx.set_item(
                        &cx.get("diffusers_unet_map")?,
                        &cx.method(
                            &text("up_blocks.{}.attentions.{}.{}"),
                            "format",
                            vec![cx.get("x")?, cx.get("i")?, cx.get("b")?],
                            vec![],
                        )?,
                        cx.method(
                            &text("output_blocks.{}.1.{}"),
                            "format",
                            vec![cx.get("n")?, cx.get("b")?],
                            vec![],
                        )?,
                    )?;
                }
                // comfy/utils.py:427
                for detector_item in
                    cx.iter(&cx.function("range", vec![cx.get("num_transformers")?], vec![])?)?
                {
                    cx.set("t", detector_item); // comfy/utils.py:428
                    for detector_item in cx.iter(&cx.get("TRANSFORMER_BLOCKS")?)? {
                        cx.set("b", detector_item); // comfy/utils.py:429
                        cx.set_item(
                            &cx.get("diffusers_unet_map")?,
                            &cx.method(
                                &text("up_blocks.{}.attentions.{}.transformer_blocks.{}.{}"),
                                "format",
                                vec![cx.get("x")?, cx.get("i")?, cx.get("t")?, cx.get("b")?],
                                vec![],
                            )?,
                            cx.method(
                                &text("output_blocks.{}.1.transformer_blocks.{}.{}"),
                                "format",
                                vec![cx.get("n")?, cx.get("t")?, cx.get("b")?],
                                vec![],
                            )?,
                        )?;
                    }
                }
            }
            // comfy/utils.py:430
            if cx.truth(&{
                let detector_tmp_8 = cx.get("i")?;
                let detector_tmp_9 = cx.binary("Sub", &cx.get("l")?, &integer(1_i128)?)?;
                cx.compare("Eq", &detector_tmp_8, &detector_tmp_9)?
            })? {
                // comfy/utils.py:431
                for detector_item in cx.iter(&list(vec![text("weight"), text("bias")]))? {
                    cx.set("k", detector_item); // comfy/utils.py:432
                    cx.set_item(
                        &cx.get("diffusers_unet_map")?,
                        &cx.method(
                            &text("up_blocks.{}.upsamplers.0.conv.{}"),
                            "format",
                            vec![cx.get("x")?, cx.get("k")?],
                            vec![],
                        )?,
                        cx.method(
                            &text("output_blocks.{}.{}.conv.{}"),
                            "format",
                            vec![cx.get("n")?, cx.get("c")?, cx.get("k")?],
                            vec![],
                        )?,
                    )?;
                }
            }
            // comfy/utils.py:433
            cx.set("n", cx.binary("Add", &cx.get("n")?, &integer(1_i128)?)?);
        }
    }
    // comfy/utils.py:435
    for detector_item in cx.iter(&cx.get("UNET_MAP_BASIC")?)? {
        cx.set("k", detector_item); // comfy/utils.py:436
        cx.set_item(
            &cx.get("diffusers_unet_map")?,
            &cx.index(&cx.get("k")?, &integer(1_i128)?)?,
            cx.index(&cx.get("k")?, &integer(0_i128)?)?,
        )?;
    }
    cx.get("diffusers_unet_map")
}
