struct NaLoop { iv: Value, arguments: [Value; 3], results: [Value; 3], origins: [String; 3], location: String }
fn nested_name(name:&str,location:&str)->String{format!("loc(\"{name}\"({}))",location.strip_prefix("loc(").unwrap().strip_suffix(')').unwrap())}
impl Builder {
fn na_full(&mut self,shape:Vec<u32>,value:Value,name:&str,p:(u32,u32))->Value{{let scalar=self.materialize(&value,Ty::Float(Element::F32),&[],name,p);self.broadcast_to(scalar,&shape,name,p)}}
fn na_zeros(&mut self,shape:Vec<u32>,stdpath:&str,name:&str,p:(u32,u32))->Value{
 let function=format!("triton.language.standard.zeros__T{}T_cfp32",shape.iter().map(|n|format!("c{n}")).collect::<Vec<_>>().join("_"));
 let ty=format!("tensor<{}xf32>",shape.iter().map(u32::to_string).collect::<Vec<_>>().join("x"));let path=serde_json::to_string(stdpath).unwrap();
 if !self.helpers.iter().any(|(n,_)|n==&function){self.helpers.push((function.clone(),format!(r#"  tt.func private @{function}() -> {ty} attributes {{noinline = false}} {{
    %zero = arith.constant 0.000000e+00 : f32 loc({path}:129:31)
    %zeros = arith.constant dense<0.000000e+00> : {ty} loc({path}:129:31)
    tt.return %zeros : {ty} loc({path}:129:11)
  ^bb1:
    %poison = ub.poison : {ty} loc({path}:129:4)
    tt.return %poison : {ty} loc({path}:129:4)
  }} loc({path}:120:0)
"#)));}
 self.emit_shape(format!("tt.call @{function}() : () -> {ty}"),Ty::Float(Element::F32),&shape,name,p)
}
fn na_trans(&mut self,v:Value,name:&str,p:(u32,u32))->Value{let shape=[v.shape[1],v.shape[0]];self.emit_shape(format!("tt.trans {} {{order = array<i32: 1, 0>}} : {} -> {}",v.id,v.typed(self.block),v.typed_shape(&shape)),v.ty,&shape,name,p)}
fn na_dot(&mut self,a:Value,b:Value,ieee:bool,name:&str,p:(u32,u32))->Value{let shape=[a.shape[0],b.shape[1]];let zero=self.materialize(&Value::number(0.),Ty::Float(Element::F32),&[],name,p);let zero=self.broadcast_to(zero,&shape,name,p);self.emit_shape(format!("tt.dot {}, {}, {}, inputPrecision = {} : {} * {} -> {}",a.id,b.id,zero.id,if ieee{"ieee"}else{"tf32"},a.typed(self.block),b.typed(self.block),zero.typed(self.block)),Ty::Float(Element::F32),&shape,name,p)}
fn na_reduce(&mut self,input:Value,sum:bool,stdpath:&str,name:&str,p:(u32,u32))->Value{
 let path=serde_json::to_string(stdpath).unwrap();let shape=[input.shape[0]];let ty=input.typed_shape(&shape);let vector=input.typed(self.block);let encoded=input.shape.iter().map(u32::to_string).collect::<Vec<_>>().join("_");
 let (operation,suffix,definition,reduce_at,return_at,dead_at,combine,combine_definition,combine_at,combine_return,combine_dead,arithmetic)=if sum{("sum","c1_cFalse_cNone",287,(293,36),(293,11),(293,4),"_sum_combine",262,(263,15),(263,11),(263,4),"addf")}else{("max","c1_cFalse_cTrue_cFalse",177,(191,40),(191,15),(179,4),"_elementwise_max",169,(170,27),(170,11),(170,4),"maxnumf")};
 let function=format!("triton.language.standard.{operation}__fp32S{encoded}S_{suffix}");let combine=format!("triton.language.standard.{combine}__fp32_fp32");
 if !self.helpers.iter().any(|(n,_)|n==&function){let (rl,rc)=reduce_at;let (tl,tc)=return_at;let (dl,dc)=dead_at;let (cl,cc)=combine_at;let (ctl,ctc)=combine_return;let (cdl,cdc)=combine_dead;self.helpers.push((function.clone(),format!(r#"  tt.func private @{function}(%input: {vector} loc("input"({path}:{definition}:0))) -> {ty} attributes {{noinline = false}} {{
    %reduced = "tt.reduce"(%input) <{{axis = 1 : i32}}> ({{
    ^bb0(%a: f32 loc(unknown), %b: f32 loc(unknown)):
      %combined = tt.call @{combine}(%a, %b) : (f32, f32) -> f32 loc({path}:{rl}:{rc})
      tt.reduce.return %combined : f32 loc({path}:{rl}:{rc})
    }}) : ({vector}) -> {ty} loc({path}:{rl}:{rc})
    tt.return %reduced : {ty} loc({path}:{tl}:{tc})
  ^bb1:
    %poison = ub.poison : {ty} loc({path}:{dl}:{dc})
    tt.return %poison : {ty} loc({path}:{dl}:{dc})
  }} loc({path}:{definition}:0)
  tt.func private @{combine}(%a: f32 loc("a"({path}:{combine_definition}:0)), %b: f32 loc("b"({path}:{combine_definition}:0))) -> f32 attributes {{noinline = false}} {{
    %combined = arith.{arithmetic} %a, %b : f32 loc({path}:{cl}:{cc})
    tt.return %combined : f32 loc({path}:{ctl}:{ctc})
  ^bb1:
    %poison = ub.poison : f32 loc({path}:{cdl}:{cdc})
    tt.return %poison : f32 loc({path}:{cdl}:{cdc})
  }} loc({path}:{combine_definition}:0)
"#)));}
 self.emit_shape(format!("tt.call @{function}({}) : ({vector}) -> {ty}",input.id),input.ty,&shape,name,p)
}
fn na_begin_loop(&mut self,bounds:[Value;3],iv_name:&str,initial:[Value;3],origins:&[String;3],p:(u32,u32))->NaLoop{
 let mut bounds=bounds.into_iter().map(|v|self.tensor(v,"",p)).collect::<Vec<_>>();let iv_type=if bounds.iter().any(|v|v.ty==Ty::I64){Ty::I64}else{Ty::I32};
 for v in &mut bounds{*v=if v.ty==iv_type{self.emit(format!("arith.bitcast {} : {} to {}",v.id,iv_type.ir(),iv_type.ir()),iv_type,false,"",p)}else{self.cast(v.clone(),iv_type,"",p)};}
 self.emit(format!("ub.poison : {}",iv_type.ir()),iv_type,false,"",p);
 let id=self.next;self.next+=1;
 let iv=Value{id:format!("%iv{id}"),ty:iv_type,shape:vec![],literal:None,float_literal:None,constexpr:false};
 let ids=initial.iter().map(|v|v.id.as_str()).collect::<Vec<_>>().join(", ");let names=["m_i","l_i","acc"];
 let origins=std::array::from_fn(|i|nested_name(names[i],&origins[i]));
 writeln!(self.body,"    %loop{id}:3 = \"scf.for\"({}, {}, {}, {ids}) ({{\n    ^bb0({}: {} {},",bounds[0].id,bounds[1].id,bounds[2].id,iv.id,iv_type.ir(),self.loc(iv_name,p)).unwrap();
 let arguments=std::array::from_fn(|i|{let mut v=initial[i].clone();v.id=format!("%carried{id}_{i}");writeln!(self.body,"      {}: {} {}{}",v.id,v.typed(self.block),origins[i],if i==2{"):"}else{","}).unwrap();v});
 let results=std::array::from_fn(|i|{let mut v=initial[i].clone();v.id=format!("%loop{id}#{i}");v});let mut location=self.loc("",p);for n in names{location=nested_name(n,&location);}
 NaLoop{iv,arguments,results,origins,location}
}
fn na_end_loop(&mut self,frame:&NaLoop,values:[Value;3],p:(u32,u32)){
 let ids=values.iter().map(|v|v.id.as_str()).collect::<Vec<_>>().join(", ");let types=values.iter().map(|v|v.typed(self.block)).collect::<Vec<_>>().join(", ");self.line(format!("scf.yield {ids} : {types}"),"",p);writeln!(self.body,"    }}) : ({}, {}, {}, {types}) -> ({types}) {}",frame.iv.ty.ir(),frame.iv.ty.ir(),frame.iv.ty.ir(),frame.location).unwrap();
}
}
