// Generated from locked comfy/model_detection.py; edit the generator.
// Source SHA256: b0f615387a1808eb01edf92bc301bb79e02c42112b566cf6826b67c45e7d71ed
use super::detector_values::*;
use crate::{native::Error, patching::options::Options};
pub(super) fn run(mut cx: Context) -> Result<Options, Error> {
    // comfy/model_detection.py:1439
    if cx.truth(&{
        let detector_tmp_1 = text("conv_in.weight");
        let detector_tmp_2 = cx.get("state_dict")?;
        cx.compare("NotIn", &detector_tmp_1, &detector_tmp_2)?
    })? {
        // comfy/model_detection.py:1440
        return Ok(none());
    }
    // comfy/model_detection.py:1442
    cx.set("match", cx.mapping(vec![])?);
    // comfy/model_detection.py:1443
    cx.set("transformer_depth", list(vec![]));
    // comfy/model_detection.py:1445
    cx.set("attn_res", integer(1_i128)?);
    // comfy/model_detection.py:1446
    cx.set(
        "down_blocks",
        cx.function(
            "count_blocks",
            vec![cx.get("state_dict")?, text("down_blocks.{}")],
            vec![],
        )?,
    );
    // comfy/model_detection.py:1447
    for detector_item in cx.iter(&cx.function("range", vec![cx.get("down_blocks")?], vec![])?)? {
        cx.set("i", detector_item); // comfy/model_detection.py:1448
        cx.set(
            "attn_blocks",
            cx.function(
                "count_blocks",
                vec![
                    cx.get("state_dict")?,
                    cx.binary(
                        "Add",
                        &cx.method(
                            &text("down_blocks.{}.attentions."),
                            "format",
                            vec![cx.get("i")?],
                            vec![],
                        )?,
                        &text("{}"),
                    )?,
                ],
                vec![],
            )?,
        );
        // comfy/model_detection.py:1449
        cx.set(
            "res_blocks",
            cx.function(
                "count_blocks",
                vec![
                    cx.get("state_dict")?,
                    cx.binary(
                        "Add",
                        &cx.method(
                            &text("down_blocks.{}.resnets."),
                            "format",
                            vec![cx.get("i")?],
                            vec![],
                        )?,
                        &text("{}"),
                    )?,
                ],
                vec![],
            )?,
        );
        // comfy/model_detection.py:1450
        for detector_item in
            cx.iter(&cx.function("range", vec![cx.get("attn_blocks")?], vec![])?)?
        {
            cx.set("ab", detector_item); // comfy/model_detection.py:1451
            cx.set(
                "transformer_count",
                cx.function(
                    "count_blocks",
                    vec![
                        cx.get("state_dict")?,
                        cx.binary(
                            "Add",
                            &cx.method(
                                &text("down_blocks.{}.attentions.{}.transformer_blocks."),
                                "format",
                                vec![cx.get("i")?, cx.get("ab")?],
                                vec![],
                            )?,
                            &text("{}"),
                        )?,
                    ],
                    vec![],
                )?,
            );
            // comfy/model_detection.py:1452
            cx.method(
                &cx.get("transformer_depth")?,
                "append",
                vec![cx.get("transformer_count")?],
                vec![],
            )?;
            // comfy/model_detection.py:1453
            if cx.truth(&{
                let detector_tmp_3 = cx.get("transformer_count")?;
                let detector_tmp_4 = integer(0_i128)?;
                cx.compare("Gt", &detector_tmp_3, &detector_tmp_4)?
            })? {
                // comfy/model_detection.py:1454
                cx.set_item(&cx.get("match")?,&text("context_dim"),cx.index(&cx.attribute(&cx.index(&cx.get("state_dict")?,&cx.method(&text("down_blocks.{}.attentions.{}.transformer_blocks.0.attn2.to_k.weight"),"format",vec![cx.get("i")?,cx.get("ab")?],vec![])?)?,"shape")?,&integer(1_i128)?)?)?;
            }
        }
        // comfy/model_detection.py:1456
        cx.set(
            "attn_res",
            cx.binary("Mult", &cx.get("attn_res")?, &integer(2_i128)?)?,
        );
        // comfy/model_detection.py:1457
        if cx.truth(&{
            let detector_tmp_5 = cx.get("attn_blocks")?;
            let detector_tmp_6 = integer(0_i128)?;
            cx.compare("Eq", &detector_tmp_5, &detector_tmp_6)?
        })? {
            // comfy/model_detection.py:1458
            for detector_item in
                cx.iter(&cx.function("range", vec![cx.get("res_blocks")?], vec![])?)?
            {
                cx.set("i", detector_item); // comfy/model_detection.py:1459
                cx.method(
                    &cx.get("transformer_depth")?,
                    "append",
                    vec![integer(0_i128)?],
                    vec![],
                )?;
            }
        }
    }
    // comfy/model_detection.py:1461
    cx.set_item(
        &cx.get("match")?,
        &text("transformer_depth"),
        cx.get("transformer_depth")?,
    )?;
    // comfy/model_detection.py:1463
    cx.set_item(
        &cx.get("match")?,
        &text("model_channels"),
        cx.index(
            &cx.attribute(
                &cx.index(&cx.get("state_dict")?, &text("conv_in.weight"))?,
                "shape",
            )?,
            &integer(0_i128)?,
        )?,
    )?;
    // comfy/model_detection.py:1464
    cx.set_item(
        &cx.get("match")?,
        &text("in_channels"),
        cx.index(
            &cx.attribute(
                &cx.index(&cx.get("state_dict")?, &text("conv_in.weight"))?,
                "shape",
            )?,
            &integer(1_i128)?,
        )?,
    )?;
    // comfy/model_detection.py:1465
    cx.set_item(&cx.get("match")?, &text("adm_in_channels"), none())?;
    // comfy/model_detection.py:1466
    if cx.truth(&{
        let detector_tmp_7 = text("class_embedding.linear_1.weight");
        let detector_tmp_8 = cx.get("state_dict")?;
        cx.compare("In", &detector_tmp_7, &detector_tmp_8)?
    })? {
        // comfy/model_detection.py:1467
        cx.set_item(
            &cx.get("match")?,
            &text("adm_in_channels"),
            cx.index(
                &cx.attribute(
                    &cx.index(
                        &cx.get("state_dict")?,
                        &text("class_embedding.linear_1.weight"),
                    )?,
                    "shape",
                )?,
                &integer(1_i128)?,
            )?,
        )?;
    } else {
        // comfy/model_detection.py:1468
        if cx.truth(&{
            let detector_tmp_9 = text("add_embedding.linear_1.weight");
            let detector_tmp_10 = cx.get("state_dict")?;
            cx.compare("In", &detector_tmp_9, &detector_tmp_10)?
        })? {
            // comfy/model_detection.py:1469
            cx.set_item(
                &cx.get("match")?,
                &text("adm_in_channels"),
                cx.index(
                    &cx.attribute(
                        &cx.index(
                            &cx.get("state_dict")?,
                            &text("add_embedding.linear_1.weight"),
                        )?,
                        "shape",
                    )?,
                    &integer(1_i128)?,
                )?,
            )?;
        }
    }
    // comfy/model_detection.py:1471
    cx.set(
        "SDXL",
        cx.mapping(vec![
            (text("use_checkpoint"), boolean(false)),
            (text("image_size"), integer(32_i128)?),
            (text("out_channels"), integer(4_i128)?),
            (text("use_spatial_transformer"), boolean(true)),
            (text("legacy"), boolean(false)),
            (text("num_classes"), text("sequential")),
            (text("adm_in_channels"), integer(2816_i128)?),
            (text("dtype"), cx.get("dtype")?),
            (text("in_channels"), integer(4_i128)?),
            (text("model_channels"), integer(320_i128)?),
            (
                text("num_res_blocks"),
                list(vec![integer(2_i128)?, integer(2_i128)?, integer(2_i128)?]),
            ),
            (
                text("transformer_depth"),
                list(vec![
                    integer(0_i128)?,
                    integer(0_i128)?,
                    integer(2_i128)?,
                    integer(2_i128)?,
                    integer(10_i128)?,
                    integer(10_i128)?,
                ]),
            ),
            (
                text("channel_mult"),
                list(vec![integer(1_i128)?, integer(2_i128)?, integer(4_i128)?]),
            ),
            (text("transformer_depth_middle"), integer(10_i128)?),
            (text("use_linear_in_transformer"), boolean(true)),
            (text("context_dim"), integer(2048_i128)?),
            (text("num_head_channels"), integer(64_i128)?),
            (
                text("transformer_depth_output"),
                list(vec![
                    integer(0_i128)?,
                    integer(0_i128)?,
                    integer(0_i128)?,
                    integer(2_i128)?,
                    integer(2_i128)?,
                    integer(2_i128)?,
                    integer(10_i128)?,
                    integer(10_i128)?,
                    integer(10_i128)?,
                ]),
            ),
            (text("use_temporal_attention"), boolean(false)),
            (text("use_temporal_resblock"), boolean(false)),
        ])?,
    );
    // comfy/model_detection.py:1477
    cx.set(
        "SDXL_refiner",
        cx.mapping(vec![
            (text("use_checkpoint"), boolean(false)),
            (text("image_size"), integer(32_i128)?),
            (text("out_channels"), integer(4_i128)?),
            (text("use_spatial_transformer"), boolean(true)),
            (text("legacy"), boolean(false)),
            (text("num_classes"), text("sequential")),
            (text("adm_in_channels"), integer(2560_i128)?),
            (text("dtype"), cx.get("dtype")?),
            (text("in_channels"), integer(4_i128)?),
            (text("model_channels"), integer(384_i128)?),
            (
                text("num_res_blocks"),
                list(vec![
                    integer(2_i128)?,
                    integer(2_i128)?,
                    integer(2_i128)?,
                    integer(2_i128)?,
                ]),
            ),
            (
                text("transformer_depth"),
                list(vec![
                    integer(0_i128)?,
                    integer(0_i128)?,
                    integer(4_i128)?,
                    integer(4_i128)?,
                    integer(4_i128)?,
                    integer(4_i128)?,
                    integer(0_i128)?,
                    integer(0_i128)?,
                ]),
            ),
            (
                text("channel_mult"),
                list(vec![
                    integer(1_i128)?,
                    integer(2_i128)?,
                    integer(4_i128)?,
                    integer(4_i128)?,
                ]),
            ),
            (text("transformer_depth_middle"), integer(4_i128)?),
            (text("use_linear_in_transformer"), boolean(true)),
            (text("context_dim"), integer(1280_i128)?),
            (text("num_head_channels"), integer(64_i128)?),
            (
                text("transformer_depth_output"),
                list(vec![
                    integer(0_i128)?,
                    integer(0_i128)?,
                    integer(0_i128)?,
                    integer(4_i128)?,
                    integer(4_i128)?,
                    integer(4_i128)?,
                    integer(4_i128)?,
                    integer(4_i128)?,
                    integer(4_i128)?,
                    integer(0_i128)?,
                    integer(0_i128)?,
                    integer(0_i128)?,
                ]),
            ),
            (text("use_temporal_attention"), boolean(false)),
            (text("use_temporal_resblock"), boolean(false)),
        ])?,
    );
    // comfy/model_detection.py:1483
    cx.set(
        "SD21",
        cx.mapping(vec![
            (text("use_checkpoint"), boolean(false)),
            (text("image_size"), integer(32_i128)?),
            (text("out_channels"), integer(4_i128)?),
            (text("use_spatial_transformer"), boolean(true)),
            (text("legacy"), boolean(false)),
            (text("adm_in_channels"), none()),
            (text("dtype"), cx.get("dtype")?),
            (text("in_channels"), integer(4_i128)?),
            (text("model_channels"), integer(320_i128)?),
            (
                text("num_res_blocks"),
                list(vec![
                    integer(2_i128)?,
                    integer(2_i128)?,
                    integer(2_i128)?,
                    integer(2_i128)?,
                ]),
            ),
            (
                text("transformer_depth"),
                list(vec![
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(0_i128)?,
                    integer(0_i128)?,
                ]),
            ),
            (
                text("channel_mult"),
                list(vec![
                    integer(1_i128)?,
                    integer(2_i128)?,
                    integer(4_i128)?,
                    integer(4_i128)?,
                ]),
            ),
            (text("transformer_depth_middle"), integer(1_i128)?),
            (text("use_linear_in_transformer"), boolean(true)),
            (text("context_dim"), integer(1024_i128)?),
            (text("num_head_channels"), integer(64_i128)?),
            (
                text("transformer_depth_output"),
                list(vec![
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(0_i128)?,
                    integer(0_i128)?,
                    integer(0_i128)?,
                ]),
            ),
            (text("use_temporal_attention"), boolean(false)),
            (text("use_temporal_resblock"), boolean(false)),
        ])?,
    );
    // comfy/model_detection.py:1489
    cx.set(
        "SD21_uncliph",
        cx.mapping(vec![
            (text("use_checkpoint"), boolean(false)),
            (text("image_size"), integer(32_i128)?),
            (text("out_channels"), integer(4_i128)?),
            (text("use_spatial_transformer"), boolean(true)),
            (text("legacy"), boolean(false)),
            (text("num_classes"), text("sequential")),
            (text("adm_in_channels"), integer(2048_i128)?),
            (text("dtype"), cx.get("dtype")?),
            (text("in_channels"), integer(4_i128)?),
            (text("model_channels"), integer(320_i128)?),
            (
                text("num_res_blocks"),
                list(vec![
                    integer(2_i128)?,
                    integer(2_i128)?,
                    integer(2_i128)?,
                    integer(2_i128)?,
                ]),
            ),
            (
                text("transformer_depth"),
                list(vec![
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(0_i128)?,
                    integer(0_i128)?,
                ]),
            ),
            (
                text("channel_mult"),
                list(vec![
                    integer(1_i128)?,
                    integer(2_i128)?,
                    integer(4_i128)?,
                    integer(4_i128)?,
                ]),
            ),
            (text("transformer_depth_middle"), integer(1_i128)?),
            (text("use_linear_in_transformer"), boolean(true)),
            (text("context_dim"), integer(1024_i128)?),
            (text("num_head_channels"), integer(64_i128)?),
            (
                text("transformer_depth_output"),
                list(vec![
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(0_i128)?,
                    integer(0_i128)?,
                    integer(0_i128)?,
                ]),
            ),
            (text("use_temporal_attention"), boolean(false)),
            (text("use_temporal_resblock"), boolean(false)),
        ])?,
    );
    // comfy/model_detection.py:1495
    cx.set(
        "SD21_unclipl",
        cx.mapping(vec![
            (text("use_checkpoint"), boolean(false)),
            (text("image_size"), integer(32_i128)?),
            (text("out_channels"), integer(4_i128)?),
            (text("use_spatial_transformer"), boolean(true)),
            (text("legacy"), boolean(false)),
            (text("num_classes"), text("sequential")),
            (text("adm_in_channels"), integer(1536_i128)?),
            (text("dtype"), cx.get("dtype")?),
            (text("in_channels"), integer(4_i128)?),
            (text("model_channels"), integer(320_i128)?),
            (
                text("num_res_blocks"),
                list(vec![
                    integer(2_i128)?,
                    integer(2_i128)?,
                    integer(2_i128)?,
                    integer(2_i128)?,
                ]),
            ),
            (
                text("transformer_depth"),
                list(vec![
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(0_i128)?,
                    integer(0_i128)?,
                ]),
            ),
            (
                text("channel_mult"),
                list(vec![
                    integer(1_i128)?,
                    integer(2_i128)?,
                    integer(4_i128)?,
                    integer(4_i128)?,
                ]),
            ),
            (text("transformer_depth_middle"), integer(1_i128)?),
            (text("use_linear_in_transformer"), boolean(true)),
            (text("context_dim"), integer(1024_i128)?),
            (text("num_head_channels"), integer(64_i128)?),
            (
                text("transformer_depth_output"),
                list(vec![
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(0_i128)?,
                    integer(0_i128)?,
                    integer(0_i128)?,
                ]),
            ),
            (text("use_temporal_attention"), boolean(false)),
            (text("use_temporal_resblock"), boolean(false)),
        ])?,
    );
    // comfy/model_detection.py:1501
    cx.set(
        "SD15",
        cx.mapping(vec![
            (text("use_checkpoint"), boolean(false)),
            (text("image_size"), integer(32_i128)?),
            (text("out_channels"), integer(4_i128)?),
            (text("use_spatial_transformer"), boolean(true)),
            (text("legacy"), boolean(false)),
            (text("adm_in_channels"), none()),
            (text("dtype"), cx.get("dtype")?),
            (text("in_channels"), integer(4_i128)?),
            (text("model_channels"), integer(320_i128)?),
            (
                text("num_res_blocks"),
                list(vec![
                    integer(2_i128)?,
                    integer(2_i128)?,
                    integer(2_i128)?,
                    integer(2_i128)?,
                ]),
            ),
            (
                text("transformer_depth"),
                list(vec![
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(0_i128)?,
                    integer(0_i128)?,
                ]),
            ),
            (
                text("channel_mult"),
                list(vec![
                    integer(1_i128)?,
                    integer(2_i128)?,
                    integer(4_i128)?,
                    integer(4_i128)?,
                ]),
            ),
            (text("transformer_depth_middle"), integer(1_i128)?),
            (text("use_linear_in_transformer"), boolean(false)),
            (text("context_dim"), integer(768_i128)?),
            (text("num_heads"), integer(8_i128)?),
            (
                text("transformer_depth_output"),
                list(vec![
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(0_i128)?,
                    integer(0_i128)?,
                    integer(0_i128)?,
                ]),
            ),
            (text("use_temporal_attention"), boolean(false)),
            (text("use_temporal_resblock"), boolean(false)),
        ])?,
    );
    // comfy/model_detection.py:1507
    cx.set(
        "SDXL_mid_cnet",
        cx.mapping(vec![
            (text("use_checkpoint"), boolean(false)),
            (text("image_size"), integer(32_i128)?),
            (text("out_channels"), integer(4_i128)?),
            (text("use_spatial_transformer"), boolean(true)),
            (text("legacy"), boolean(false)),
            (text("num_classes"), text("sequential")),
            (text("adm_in_channels"), integer(2816_i128)?),
            (text("dtype"), cx.get("dtype")?),
            (text("in_channels"), integer(4_i128)?),
            (text("model_channels"), integer(320_i128)?),
            (
                text("num_res_blocks"),
                list(vec![integer(2_i128)?, integer(2_i128)?, integer(2_i128)?]),
            ),
            (
                text("transformer_depth"),
                list(vec![
                    integer(0_i128)?,
                    integer(0_i128)?,
                    integer(0_i128)?,
                    integer(0_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                ]),
            ),
            (
                text("channel_mult"),
                list(vec![integer(1_i128)?, integer(2_i128)?, integer(4_i128)?]),
            ),
            (text("transformer_depth_middle"), integer(1_i128)?),
            (text("use_linear_in_transformer"), boolean(true)),
            (text("context_dim"), integer(2048_i128)?),
            (text("num_head_channels"), integer(64_i128)?),
            (
                text("transformer_depth_output"),
                list(vec![
                    integer(0_i128)?,
                    integer(0_i128)?,
                    integer(0_i128)?,
                    integer(0_i128)?,
                    integer(0_i128)?,
                    integer(0_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                ]),
            ),
            (text("use_temporal_attention"), boolean(false)),
            (text("use_temporal_resblock"), boolean(false)),
        ])?,
    );
    // comfy/model_detection.py:1513
    cx.set(
        "SDXL_small_cnet",
        cx.mapping(vec![
            (text("use_checkpoint"), boolean(false)),
            (text("image_size"), integer(32_i128)?),
            (text("out_channels"), integer(4_i128)?),
            (text("use_spatial_transformer"), boolean(true)),
            (text("legacy"), boolean(false)),
            (text("num_classes"), text("sequential")),
            (text("adm_in_channels"), integer(2816_i128)?),
            (text("dtype"), cx.get("dtype")?),
            (text("in_channels"), integer(4_i128)?),
            (text("model_channels"), integer(320_i128)?),
            (
                text("num_res_blocks"),
                list(vec![integer(2_i128)?, integer(2_i128)?, integer(2_i128)?]),
            ),
            (
                text("transformer_depth"),
                list(vec![
                    integer(0_i128)?,
                    integer(0_i128)?,
                    integer(0_i128)?,
                    integer(0_i128)?,
                    integer(0_i128)?,
                    integer(0_i128)?,
                ]),
            ),
            (
                text("channel_mult"),
                list(vec![integer(1_i128)?, integer(2_i128)?, integer(4_i128)?]),
            ),
            (text("transformer_depth_middle"), integer(0_i128)?),
            (text("use_linear_in_transformer"), boolean(true)),
            (text("num_head_channels"), integer(64_i128)?),
            (text("context_dim"), integer(1_i128)?),
            (
                text("transformer_depth_output"),
                list(vec![
                    integer(0_i128)?,
                    integer(0_i128)?,
                    integer(0_i128)?,
                    integer(0_i128)?,
                    integer(0_i128)?,
                    integer(0_i128)?,
                    integer(0_i128)?,
                    integer(0_i128)?,
                    integer(0_i128)?,
                ]),
            ),
            (text("use_temporal_attention"), boolean(false)),
            (text("use_temporal_resblock"), boolean(false)),
        ])?,
    );
    // comfy/model_detection.py:1519
    cx.set(
        "SDXL_diffusers_inpaint",
        cx.mapping(vec![
            (text("use_checkpoint"), boolean(false)),
            (text("image_size"), integer(32_i128)?),
            (text("out_channels"), integer(4_i128)?),
            (text("use_spatial_transformer"), boolean(true)),
            (text("legacy"), boolean(false)),
            (text("num_classes"), text("sequential")),
            (text("adm_in_channels"), integer(2816_i128)?),
            (text("dtype"), cx.get("dtype")?),
            (text("in_channels"), integer(9_i128)?),
            (text("model_channels"), integer(320_i128)?),
            (
                text("num_res_blocks"),
                list(vec![integer(2_i128)?, integer(2_i128)?, integer(2_i128)?]),
            ),
            (
                text("transformer_depth"),
                list(vec![
                    integer(0_i128)?,
                    integer(0_i128)?,
                    integer(2_i128)?,
                    integer(2_i128)?,
                    integer(10_i128)?,
                    integer(10_i128)?,
                ]),
            ),
            (
                text("channel_mult"),
                list(vec![integer(1_i128)?, integer(2_i128)?, integer(4_i128)?]),
            ),
            (text("transformer_depth_middle"), integer(10_i128)?),
            (text("use_linear_in_transformer"), boolean(true)),
            (text("context_dim"), integer(2048_i128)?),
            (text("num_head_channels"), integer(64_i128)?),
            (
                text("transformer_depth_output"),
                list(vec![
                    integer(0_i128)?,
                    integer(0_i128)?,
                    integer(0_i128)?,
                    integer(2_i128)?,
                    integer(2_i128)?,
                    integer(2_i128)?,
                    integer(10_i128)?,
                    integer(10_i128)?,
                    integer(10_i128)?,
                ]),
            ),
            (text("use_temporal_attention"), boolean(false)),
            (text("use_temporal_resblock"), boolean(false)),
        ])?,
    );
    // comfy/model_detection.py:1525
    cx.set(
        "SDXL_diffusers_ip2p",
        cx.mapping(vec![
            (text("use_checkpoint"), boolean(false)),
            (text("image_size"), integer(32_i128)?),
            (text("out_channels"), integer(4_i128)?),
            (text("use_spatial_transformer"), boolean(true)),
            (text("legacy"), boolean(false)),
            (text("num_classes"), text("sequential")),
            (text("adm_in_channels"), integer(2816_i128)?),
            (text("dtype"), cx.get("dtype")?),
            (text("in_channels"), integer(8_i128)?),
            (text("model_channels"), integer(320_i128)?),
            (
                text("num_res_blocks"),
                list(vec![integer(2_i128)?, integer(2_i128)?, integer(2_i128)?]),
            ),
            (
                text("transformer_depth"),
                list(vec![
                    integer(0_i128)?,
                    integer(0_i128)?,
                    integer(2_i128)?,
                    integer(2_i128)?,
                    integer(10_i128)?,
                    integer(10_i128)?,
                ]),
            ),
            (
                text("channel_mult"),
                list(vec![integer(1_i128)?, integer(2_i128)?, integer(4_i128)?]),
            ),
            (text("transformer_depth_middle"), integer(10_i128)?),
            (text("use_linear_in_transformer"), boolean(true)),
            (text("context_dim"), integer(2048_i128)?),
            (text("num_head_channels"), integer(64_i128)?),
            (
                text("transformer_depth_output"),
                list(vec![
                    integer(0_i128)?,
                    integer(0_i128)?,
                    integer(0_i128)?,
                    integer(2_i128)?,
                    integer(2_i128)?,
                    integer(2_i128)?,
                    integer(10_i128)?,
                    integer(10_i128)?,
                    integer(10_i128)?,
                ]),
            ),
            (text("use_temporal_attention"), boolean(false)),
            (text("use_temporal_resblock"), boolean(false)),
        ])?,
    );
    // comfy/model_detection.py:1531
    cx.set(
        "SSD_1B",
        cx.mapping(vec![
            (text("use_checkpoint"), boolean(false)),
            (text("image_size"), integer(32_i128)?),
            (text("out_channels"), integer(4_i128)?),
            (text("use_spatial_transformer"), boolean(true)),
            (text("legacy"), boolean(false)),
            (text("num_classes"), text("sequential")),
            (text("adm_in_channels"), integer(2816_i128)?),
            (text("dtype"), cx.get("dtype")?),
            (text("in_channels"), integer(4_i128)?),
            (text("model_channels"), integer(320_i128)?),
            (
                text("num_res_blocks"),
                list(vec![integer(2_i128)?, integer(2_i128)?, integer(2_i128)?]),
            ),
            (
                text("transformer_depth"),
                list(vec![
                    integer(0_i128)?,
                    integer(0_i128)?,
                    integer(2_i128)?,
                    integer(2_i128)?,
                    integer(4_i128)?,
                    integer(4_i128)?,
                ]),
            ),
            (
                text("transformer_depth_output"),
                list(vec![
                    integer(0_i128)?,
                    integer(0_i128)?,
                    integer(0_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(2_i128)?,
                    integer(10_i128)?,
                    integer(4_i128)?,
                    integer(4_i128)?,
                ]),
            ),
            (
                text("channel_mult"),
                list(vec![integer(1_i128)?, integer(2_i128)?, integer(4_i128)?]),
            ),
            (
                text("transformer_depth_middle"),
                cx.unary("USub", &integer(1_i128)?)?,
            ),
            (text("use_linear_in_transformer"), boolean(true)),
            (text("context_dim"), integer(2048_i128)?),
            (text("num_head_channels"), integer(64_i128)?),
            (text("use_temporal_attention"), boolean(false)),
            (text("use_temporal_resblock"), boolean(false)),
        ])?,
    );
    // comfy/model_detection.py:1537
    cx.set(
        "Segmind_Vega",
        cx.mapping(vec![
            (text("use_checkpoint"), boolean(false)),
            (text("image_size"), integer(32_i128)?),
            (text("out_channels"), integer(4_i128)?),
            (text("use_spatial_transformer"), boolean(true)),
            (text("legacy"), boolean(false)),
            (text("num_classes"), text("sequential")),
            (text("adm_in_channels"), integer(2816_i128)?),
            (text("dtype"), cx.get("dtype")?),
            (text("in_channels"), integer(4_i128)?),
            (text("model_channels"), integer(320_i128)?),
            (
                text("num_res_blocks"),
                list(vec![integer(2_i128)?, integer(2_i128)?, integer(2_i128)?]),
            ),
            (
                text("transformer_depth"),
                list(vec![
                    integer(0_i128)?,
                    integer(0_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(2_i128)?,
                    integer(2_i128)?,
                ]),
            ),
            (
                text("transformer_depth_output"),
                list(vec![
                    integer(0_i128)?,
                    integer(0_i128)?,
                    integer(0_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(2_i128)?,
                    integer(2_i128)?,
                    integer(2_i128)?,
                ]),
            ),
            (
                text("channel_mult"),
                list(vec![integer(1_i128)?, integer(2_i128)?, integer(4_i128)?]),
            ),
            (
                text("transformer_depth_middle"),
                cx.unary("USub", &integer(1_i128)?)?,
            ),
            (text("use_linear_in_transformer"), boolean(true)),
            (text("context_dim"), integer(2048_i128)?),
            (text("num_head_channels"), integer(64_i128)?),
            (text("use_temporal_attention"), boolean(false)),
            (text("use_temporal_resblock"), boolean(false)),
        ])?,
    );
    // comfy/model_detection.py:1543
    cx.set(
        "KOALA_700M",
        cx.mapping(vec![
            (text("use_checkpoint"), boolean(false)),
            (text("image_size"), integer(32_i128)?),
            (text("out_channels"), integer(4_i128)?),
            (text("use_spatial_transformer"), boolean(true)),
            (text("legacy"), boolean(false)),
            (text("num_classes"), text("sequential")),
            (text("adm_in_channels"), integer(2816_i128)?),
            (text("dtype"), cx.get("dtype")?),
            (text("in_channels"), integer(4_i128)?),
            (text("model_channels"), integer(320_i128)?),
            (
                text("num_res_blocks"),
                list(vec![integer(1_i128)?, integer(1_i128)?, integer(1_i128)?]),
            ),
            (
                text("transformer_depth"),
                list(vec![integer(0_i128)?, integer(2_i128)?, integer(5_i128)?]),
            ),
            (
                text("transformer_depth_output"),
                list(vec![
                    integer(0_i128)?,
                    integer(0_i128)?,
                    integer(2_i128)?,
                    integer(2_i128)?,
                    integer(5_i128)?,
                    integer(5_i128)?,
                ]),
            ),
            (
                text("channel_mult"),
                list(vec![integer(1_i128)?, integer(2_i128)?, integer(4_i128)?]),
            ),
            (
                text("transformer_depth_middle"),
                cx.unary("USub", &integer(2_i128)?)?,
            ),
            (text("use_linear_in_transformer"), boolean(true)),
            (text("context_dim"), integer(2048_i128)?),
            (text("num_head_channels"), integer(64_i128)?),
            (text("use_temporal_attention"), boolean(false)),
            (text("use_temporal_resblock"), boolean(false)),
        ])?,
    );
    // comfy/model_detection.py:1549
    cx.set(
        "KOALA_1B",
        cx.mapping(vec![
            (text("use_checkpoint"), boolean(false)),
            (text("image_size"), integer(32_i128)?),
            (text("out_channels"), integer(4_i128)?),
            (text("use_spatial_transformer"), boolean(true)),
            (text("legacy"), boolean(false)),
            (text("num_classes"), text("sequential")),
            (text("adm_in_channels"), integer(2816_i128)?),
            (text("dtype"), cx.get("dtype")?),
            (text("in_channels"), integer(4_i128)?),
            (text("model_channels"), integer(320_i128)?),
            (
                text("num_res_blocks"),
                list(vec![integer(1_i128)?, integer(1_i128)?, integer(1_i128)?]),
            ),
            (
                text("transformer_depth"),
                list(vec![integer(0_i128)?, integer(2_i128)?, integer(6_i128)?]),
            ),
            (
                text("transformer_depth_output"),
                list(vec![
                    integer(0_i128)?,
                    integer(0_i128)?,
                    integer(2_i128)?,
                    integer(2_i128)?,
                    integer(6_i128)?,
                    integer(6_i128)?,
                ]),
            ),
            (
                text("channel_mult"),
                list(vec![integer(1_i128)?, integer(2_i128)?, integer(4_i128)?]),
            ),
            (text("transformer_depth_middle"), integer(6_i128)?),
            (text("use_linear_in_transformer"), boolean(true)),
            (text("context_dim"), integer(2048_i128)?),
            (text("num_head_channels"), integer(64_i128)?),
            (text("use_temporal_attention"), boolean(false)),
            (text("use_temporal_resblock"), boolean(false)),
        ])?,
    );
    // comfy/model_detection.py:1555
    cx.set(
        "SD09_XS",
        cx.mapping(vec![
            (text("use_checkpoint"), boolean(false)),
            (text("image_size"), integer(32_i128)?),
            (text("out_channels"), integer(4_i128)?),
            (text("use_spatial_transformer"), boolean(true)),
            (text("legacy"), boolean(false)),
            (text("adm_in_channels"), none()),
            (text("dtype"), cx.get("dtype")?),
            (text("in_channels"), integer(4_i128)?),
            (text("model_channels"), integer(320_i128)?),
            (
                text("num_res_blocks"),
                list(vec![integer(1_i128)?, integer(1_i128)?, integer(1_i128)?]),
            ),
            (
                text("transformer_depth"),
                list(vec![integer(1_i128)?, integer(1_i128)?, integer(1_i128)?]),
            ),
            (
                text("channel_mult"),
                list(vec![integer(1_i128)?, integer(2_i128)?, integer(4_i128)?]),
            ),
            (
                text("transformer_depth_middle"),
                cx.unary("USub", &integer(2_i128)?)?,
            ),
            (text("use_linear_in_transformer"), boolean(true)),
            (text("context_dim"), integer(1024_i128)?),
            (text("num_head_channels"), integer(64_i128)?),
            (
                text("transformer_depth_output"),
                list(vec![
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                ]),
            ),
            (text("use_temporal_attention"), boolean(false)),
            (text("use_temporal_resblock"), boolean(false)),
            (
                text("disable_self_attentions"),
                list(vec![boolean(true), boolean(false), boolean(false)]),
            ),
        ])?,
    );
    // comfy/model_detection.py:1561
    cx.set(
        "SD_XS",
        cx.mapping(vec![
            (text("use_checkpoint"), boolean(false)),
            (text("image_size"), integer(32_i128)?),
            (text("out_channels"), integer(4_i128)?),
            (text("use_spatial_transformer"), boolean(true)),
            (text("legacy"), boolean(false)),
            (text("adm_in_channels"), none()),
            (text("dtype"), cx.get("dtype")?),
            (text("in_channels"), integer(4_i128)?),
            (text("model_channels"), integer(320_i128)?),
            (
                text("num_res_blocks"),
                list(vec![integer(1_i128)?, integer(1_i128)?, integer(1_i128)?]),
            ),
            (
                text("transformer_depth"),
                list(vec![integer(0_i128)?, integer(1_i128)?, integer(1_i128)?]),
            ),
            (
                text("channel_mult"),
                list(vec![integer(1_i128)?, integer(2_i128)?, integer(4_i128)?]),
            ),
            (
                text("transformer_depth_middle"),
                cx.unary("USub", &integer(2_i128)?)?,
            ),
            (text("use_linear_in_transformer"), boolean(false)),
            (text("context_dim"), integer(768_i128)?),
            (text("num_head_channels"), integer(64_i128)?),
            (
                text("transformer_depth_output"),
                list(vec![
                    integer(0_i128)?,
                    integer(0_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                ]),
            ),
            (text("use_temporal_attention"), boolean(false)),
            (text("use_temporal_resblock"), boolean(false)),
        ])?,
    );
    // comfy/model_detection.py:1567
    cx.set(
        "SD15_diffusers_inpaint",
        cx.mapping(vec![
            (text("use_checkpoint"), boolean(false)),
            (text("image_size"), integer(32_i128)?),
            (text("out_channels"), integer(4_i128)?),
            (text("use_spatial_transformer"), boolean(true)),
            (text("legacy"), boolean(false)),
            (text("adm_in_channels"), none()),
            (text("dtype"), cx.get("dtype")?),
            (text("in_channels"), integer(9_i128)?),
            (text("model_channels"), integer(320_i128)?),
            (
                text("num_res_blocks"),
                list(vec![
                    integer(2_i128)?,
                    integer(2_i128)?,
                    integer(2_i128)?,
                    integer(2_i128)?,
                ]),
            ),
            (
                text("transformer_depth"),
                list(vec![
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(0_i128)?,
                    integer(0_i128)?,
                ]),
            ),
            (
                text("channel_mult"),
                list(vec![
                    integer(1_i128)?,
                    integer(2_i128)?,
                    integer(4_i128)?,
                    integer(4_i128)?,
                ]),
            ),
            (text("transformer_depth_middle"), integer(1_i128)?),
            (text("use_linear_in_transformer"), boolean(false)),
            (text("context_dim"), integer(768_i128)?),
            (text("num_heads"), integer(8_i128)?),
            (
                text("transformer_depth_output"),
                list(vec![
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(0_i128)?,
                    integer(0_i128)?,
                    integer(0_i128)?,
                ]),
            ),
            (text("use_temporal_attention"), boolean(false)),
            (text("use_temporal_resblock"), boolean(false)),
        ])?,
    );
    // comfy/model_detection.py:1573
    cx.set(
        "LotusD",
        cx.mapping(vec![
            (text("use_checkpoint"), boolean(false)),
            (text("image_size"), integer(32_i128)?),
            (text("out_channels"), integer(4_i128)?),
            (text("use_spatial_transformer"), boolean(true)),
            (text("legacy"), boolean(false)),
            (text("adm_in_channels"), integer(4_i128)?),
            (text("dtype"), cx.get("dtype")?),
            (text("in_channels"), integer(4_i128)?),
            (text("model_channels"), integer(320_i128)?),
            (
                text("num_res_blocks"),
                list(vec![
                    integer(2_i128)?,
                    integer(2_i128)?,
                    integer(2_i128)?,
                    integer(2_i128)?,
                ]),
            ),
            (
                text("transformer_depth"),
                list(vec![
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(0_i128)?,
                    integer(0_i128)?,
                ]),
            ),
            (
                text("channel_mult"),
                list(vec![
                    integer(1_i128)?,
                    integer(2_i128)?,
                    integer(4_i128)?,
                    integer(4_i128)?,
                ]),
            ),
            (text("transformer_depth_middle"), integer(1_i128)?),
            (text("use_linear_in_transformer"), boolean(true)),
            (text("context_dim"), integer(1024_i128)?),
            (text("num_head_channels"), integer(64_i128)?),
            (
                text("transformer_depth_output"),
                list(vec![
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(1_i128)?,
                    integer(0_i128)?,
                    integer(0_i128)?,
                    integer(0_i128)?,
                ]),
            ),
            (text("use_temporal_attention"), boolean(false)),
            (text("use_temporal_resblock"), boolean(false)),
        ])?,
    );
    // comfy/model_detection.py:1579
    cx.set(
        "supported_models",
        list(vec![
            cx.get("LotusD")?,
            cx.get("SDXL")?,
            cx.get("SDXL_refiner")?,
            cx.get("SD21")?,
            cx.get("SD15")?,
            cx.get("SD21_uncliph")?,
            cx.get("SD21_unclipl")?,
            cx.get("SDXL_mid_cnet")?,
            cx.get("SDXL_small_cnet")?,
            cx.get("SDXL_diffusers_inpaint")?,
            cx.get("SSD_1B")?,
            cx.get("Segmind_Vega")?,
            cx.get("KOALA_700M")?,
            cx.get("KOALA_1B")?,
            cx.get("SD09_XS")?,
            cx.get("SD_XS")?,
            cx.get("SDXL_diffusers_ip2p")?,
            cx.get("SD15_diffusers_inpaint")?,
        ]),
    );
    // comfy/model_detection.py:1581
    for detector_item in cx.iter(&cx.get("supported_models")?)? {
        cx.set("unet_config", detector_item); // comfy/model_detection.py:1582
        cx.set("matches", boolean(true));
        // comfy/model_detection.py:1583
        for detector_item in cx.iter(&cx.get("match")?)? {
            cx.set("k", detector_item); // comfy/model_detection.py:1584
            if cx.truth(&{
                let detector_tmp_11 = cx.index(&cx.get("match")?, &cx.get("k")?)?;
                let detector_tmp_12 = cx.index(&cx.get("unet_config")?, &cx.get("k")?)?;
                cx.compare("NotEq", &detector_tmp_11, &detector_tmp_12)?
            })? {
                // comfy/model_detection.py:1585
                cx.set("matches", boolean(false));
                // comfy/model_detection.py:1586
                break;
            }
        }
        // comfy/model_detection.py:1587
        if cx.truth(&cx.get("matches")?)? {
            // comfy/model_detection.py:1588
            return cx.function("convert_config", vec![cx.get("unet_config")?], vec![]);
        }
    }
    Ok(none())
}
