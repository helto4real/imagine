// Generated from the locked MXFP8 and integer helper source; no saved IR.
pub const SCALAR_NAMES:[&str;6]=["m", "n", "num_blocks", "scale_rows", "scale_cols", "padded_scale_cols"];
fn lower_main(b:&mut Builder,s:&Specialization<'_>,pointers:&[Value],scalars:&[Value]){
let x_ptr=pointers[0].clone();
let output_ptr=pointers[1].clone();
let swizzled_scales_ptr=pointers[2].clone();
let _m=scalars[0].clone();
let n=scalars[1].clone();
let num_blocks=scalars[2].clone();
let scale_rows=scalars[3].clone();
let scale_cols=scalars[4].clone();
let padded_scale_cols=scalars[5].clone();
let block_size=Value::integer(i64::from(s.block_size));let blocks_per_program=Value::integer(i64::from(s.blocks_per_program));
let t0=b.emit("tt.get_program_id x : i32".into(),Ty::I32,false,"pid_m",(620,26));
let pid_m=b.tensor(t0,"pid_m",(620,12));
let t1=b.emit("tt.get_program_id y : i32".into(),Ty::I32,false,"pid_n_base",(621,31));
let t2=b.bin("mul",t1,blocks_per_program.clone(),"pid_n_base",(621,41));
let pid_n_base=b.tensor(t2,"pid_n_base",(621,17));
let fp8_max=Value::number(f64::from_bits(4646588915539509248));
let t3=b.begin_for(i64::from(s.blocks_per_program),"block_offset",(627,30));
{
let block_offset=t3.1.clone();
let t4=b.bin("add",pid_n_base.clone(),block_offset.clone(),"pid_n",(628,29));
let pid_n=b.tensor(t4,"pid_n",(628,16));
let t5=b.bin("lt",pid_n.clone(),num_blocks.clone(),"",(630,19));
let t6=b.begin_if(t5,(630,11));
{
let t7=b.bin("mul",pid_n.clone(),block_size.clone(),"offs_n",(632,29));
let t8=b.emit(format!("tt.make_range {{end = {} : i32, start = 0 : i32}} : tensor<{}xi32>",s.block_size,s.block_size),Ty::I32,true,"offs_n",(632,55));
let t9=b.bin("add",t7,t8,"offs_n",(632,42));
let offs_n=b.tensor(t9,"offs_n",(632,21));
let t10=b.bin("lt",offs_n.clone(),n.clone(),"mask",(633,28));
let mask=b.tensor(t10,"mask",(633,19));
let t11=b.bin("mul",pid_m.clone(),n.clone(),"x_offs",(634,29));
let t12=b.bin("add",t11,offs_n.clone(),"x_offs",(634,33));
let x_offs=b.tensor(t12,"x_offs",(634,21));
let t13=b.bin("add",x_ptr.clone(),x_offs.clone(),"x",(637,32));
let t14=b.load(t13,mask.clone(),"x",(637,24));
let t15=b.cast(t14,Ty::Float(Element::F32),"x",(637,65));
let x=b.tensor(t15,"x",(637,16));
let t16=b.math("absf",x.clone(),"x_abs",(640,27));
let x_abs=b.tensor(t16,"x_abs",(640,20));
let t17=b.reduce_max(x_abs.clone(),s.standard_source_path,"max_abs",(641,29));
let max_abs=b.tensor(t17,"max_abs",(641,22));
let t18=b.bin("true_div",max_abs.clone(),fp8_max.clone(),"scale_ratio",(647,36));
let scale_ratio=b.tensor(t18,"scale_ratio",(647,26));
let t19=b.extreme("maximum",scale_ratio.clone(),Value::number(f64::from_bits(4035225266123964416)),"scale_ratio",(649,50));
let scale_ratio=b.tensor(t19,"scale_ratio",(649,26));
let t20=b.extreme("minimum",scale_ratio.clone(),Value::number(f64::from_bits(5179139571476070400)),"scale_ratio",(650,50));
let scale_ratio=b.tensor(t20,"scale_ratio",(650,26));
let t21=b.math("log2",scale_ratio.clone(),"log2_ratio",(653,33));
let log2_ratio=b.tensor(t21,"log2_ratio",(653,25));
let t22=b.math("ceil",log2_ratio.clone(),"exp_unbiased",(654,35));
let t23=b.cast(t22,Ty::I32,"exp_unbiased",(654,50));
let exp_unbiased=b.tensor(t23,"exp_unbiased",(654,27));
let t24=b.bin("add",exp_unbiased.clone(),Value::integer(127),"exp_biased",(655,40));
let exp_biased=b.tensor(t24,"exp_biased",(655,25));
let t25=b.extreme("maximum",exp_biased.clone(),Value::integer(0),"exp_biased",(658,48));
let exp_biased=b.tensor(t25,"exp_biased",(658,25));
let t26=b.extreme("minimum",exp_biased.clone(),Value::integer(254),"exp_biased",(659,48));
let exp_biased=b.tensor(t26,"exp_biased",(659,25));
let t27=b.bin("sub",exp_biased.clone(),Value::integer(127),"block_scale",(662,48));
let t28=b.cast(t27,Ty::Float(Element::F32),"block_scale",(662,56));
let t29=b.math("exp2",t28,"block_scale",(662,34));
let block_scale=b.tensor(t29,"block_scale",(662,26));
let t30=b.cdiv(scale_cols.clone(),Value::integer(4),s.standard_source_path,"n_col_blocks",(665,47));
let n_col_blocks=b.tensor(t30,"n_col_blocks",(665,27));
let t31=b.swizzle(vec![pid_m.clone(),pid_n.clone(),n_col_blocks.clone(),padded_scale_cols.clone()],s.source_path,"swizzled_offs",(667,44));
let swizzled_offs=b.tensor(t31,"swizzled_offs",(666,28));
let t32=b.bin("lt",pid_m.clone(),scale_rows.clone(),"",(669,23));
let t33=b.bin("lt",pid_n.clone(),scale_cols.clone(),"",(669,46));
let t34=b.bin("and",t32,t33,"",(669,38));
let t35=b.begin_if(t34,(669,15));
{
let t36=b.bin("add",swizzled_scales_ptr.clone(),swizzled_offs.clone(),"",(671,47));
let t37=b.cast(exp_biased.clone(),Ty::U8,"",(671,76));
b.store_optional(t36,t37,None,(671,62));
}
b.end_if(t35);
let t38=b.bin("lt",block_scale.clone(),Value::number(f64::from_bits(4158027847206421152)),"safe_scale",(675,48));
let t39=b.select(t38,Value::number(f64::from_bits(4607182418800017408)),block_scale.clone(),"safe_scale",(675,60));
let safe_scale=b.tensor(t39,"safe_scale",(675,25));
let t40=b.bin("true_div",x.clone(),safe_scale.clone(),"data_scaled",(676,30));
let data_scaled=b.tensor(t40,"data_scaled",(676,26));
let t41=b.bin("lt",block_scale.clone(),Value::number(f64::from_bits(4158027847206421152)),"data_scaled",(677,49));
let t42=b.select(t41,Value::number(f64::from_bits(0)),data_scaled.clone(),"data_scaled",(677,61));
let data_scaled=b.tensor(t42,"data_scaled",(677,26));
let t43=b.extreme("minimum",data_scaled.clone(),fp8_max.clone(),"data_clamped",(680,62));
let t44=b.extreme("maximum",t43,Value::number(-448.0),"data_clamped",(680,72));
let data_clamped=b.tensor(t44,"data_clamped",(680,27));
let t45=b.bin("mul",pid_m.clone(),n.clone(),"out_offs",(683,31));
let t46=b.bin("add",t45,offs_n.clone(),"out_offs",(683,35));
let out_offs=b.tensor(t46,"out_offs",(683,23));
let t47=b.bin("add",output_ptr.clone(),out_offs.clone(),"",(684,34));
let t48=b.cast(data_clamped.clone(),Ty::Float(Element::E4M3),"",(684,60));
b.store_optional(t47,t48,Some(mask.clone()),(684,44));
}
b.end_if(t6);
}
b.end_for(t3.0);
b.line("tt.return".into(),"",(627,4));
}
fn lower_cdiv(b:&mut Builder,values:&[Value])->Value{
let x=values[0].clone();
let div=values[1].clone();
let t49=b.bin("sub",div.clone(),Value::integer(1),"",(43,23));
let t50=b.bin("add",x.clone(),t49,"",(43,17));
let t51=b.bin("div",t50,div.clone(),"",(43,30));
let result=t51;
if result.constexpr{b.line("tt.return".into(),"",(43,11));}else{b.line(format!("tt.return {} : {}",result.id,result.typed(b.block)),"",(43,11));}
result
}
fn lower_swizzle(b:&mut Builder,values:&[Value])->Value{
let in_row=values[0].clone();
let in_col=values[1].clone();
let n_col_blocks=values[2].clone();
let padded_scale_cols=values[3].clone();
let t52=b.bin("div",in_row.clone(),Value::integer(128),"row_block",(173,26));
let row_block=b.tensor(t52,"row_block",(173,16));
let t53=b.bin("div",in_col.clone(),Value::integer(4),"col_block",(174,26));
let col_block=b.tensor(t53,"col_block",(174,16));
let t54=b.bin("rem",in_row.clone(),Value::integer(128),"in_block_row",(177,28));
let in_block_row=b.tensor(t54,"in_block_row",(177,19));
let t55=b.bin("rem",in_col.clone(),Value::integer(4),"in_block_col",(178,28));
let in_block_col=b.tensor(t55,"in_block_col",(178,19));
let t56=b.bin("div",in_block_row.clone(),Value::integer(32),"sub_block",(181,32));
let sub_block=b.tensor(t56,"sub_block",(181,16));
let t57=b.bin("rem",in_block_row.clone(),Value::integer(32),"fine_row",(182,30));
let fine_row=b.tensor(t57,"fine_row",(182,15));
let t58=b.bin("mul",row_block.clone(),n_col_blocks.clone(),"combined_block",(184,33));
let t59=b.bin("add",t58,col_block.clone(),"combined_block",(184,48));
let combined_block=b.tensor(t59,"combined_block",(184,21));
let t60=b.bin("mul",sub_block.clone(),Value::integer(4),"intermediate_col",(185,35));
let t61=b.bin("add",t60,in_block_col.clone(),"intermediate_col",(185,39));
let intermediate_col=b.tensor(t61,"intermediate_col",(185,23));
let t62=b.bin("mul",combined_block.clone(),Value::integer(512),"linear_idx",(188,34));
let t63=b.bin("mul",fine_row.clone(),Value::integer(16),"linear_idx",(188,51));
let t64=b.bin("add",t62,t63,"linear_idx",(188,40));
let t65=b.bin("add",t64,intermediate_col.clone(),"linear_idx",(188,56));
let linear_idx=b.tensor(t65,"linear_idx",(188,17));
let t66=b.bin("div",linear_idx.clone(),padded_scale_cols.clone(),"out_row",(191,28));
let out_row=b.tensor(t66,"out_row",(191,14));
let t67=b.bin("rem",linear_idx.clone(),padded_scale_cols.clone(),"out_col",(192,27));
let out_col=b.tensor(t67,"out_col",(192,14));
let t68=b.bin("mul",out_row.clone(),padded_scale_cols.clone(),"",(193,21));
let t69=b.bin("add",t68,out_col.clone(),"",(193,41));
let result=t69;
if result.constexpr{b.line("tt.return".into(),"",(193,11));}else{b.line(format!("tt.return {} : {}",result.id,result.typed(b.block)),"",(193,11));}
result
}
