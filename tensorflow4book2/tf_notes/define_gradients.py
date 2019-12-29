#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2019/12/27 19:25
# @Author  : Leslee


def apply_gradients(self,grads_and_vars,global_step=None,name=None):
    grads_and_vars = tuple(grads_and_vars)
    if not grads_and_vars:
        raise ValueError("no variables provided")
    converted_grads_and_vars = []
    for g,v in grads_and_vars:
        if g is not None:
            try:
                g = ops.convert_to_tensor_or_indexed_slices(g)
            except TypeError:
                raise TypeError("Gradients must be convert to tensor")
            if not isinstance(g,(ops.Tensor,ops.IndexedSlices)):
                raise TypeError("gradients must be a tensormIndex,or None:%s" % g)
        p = _get_processor(v)
        converted_grads_and_vars.append((g,v,p))
    converted_grads_and_vars = tuple(converted_grads_and_vars)

    var_list = [v for g,v,_ in converted_grads_and_vars if g is not None]
    if not var_list:
        raise ValueError("No gradients provided for any variable: %s." %
                       ([str(v) for _, v, _ in converted_grads_and_vars],))
    with ops.control_dependencies(None):
        self._create_slots(var_list)

    update_ops = []
    with ops.name_scope(name,self._name) as name:
        self._prepare()
        for grad,var,processor in converted_grads_and_vars:
            if grad is None:
                continue
            with ops.name_scope("update_"+var.op.name),ops.colocate_with(var):
                update_ops.append(processor.update_op(self,grad))

        if global_step is None:
            apply_updates = self._finish(update_ops,name)
        else:
            with ops.control_dependencies(global_step):
                apply_updates = state_ops.assign_add(global_step,1,name=name).op
        train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
        if apply_updates not in train_op:
            train_op.append(apply_updates)
        return apply_updates






