import torch 

def pred_confidence_filter(preds: dict, confidence_threshold: float) -> dict[str, torch.Tensor]:
        indices_to_keep = []
        for (index, item) in enumerate(preds['scores']):
            if item > confidence_threshold:
                  indices_to_keep.append(index)
            
        return {'boxes': preds['boxes'][indices_to_keep],
                'labels': preds['labels'][indices_to_keep],
                'scores': preds['scores'][indices_to_keep]}

def find_overlapping_bboxes(model_predictions) ->list[list[float,float,float,float]]:
    # separate predictions into separate classes
    water_hyacinth_predictions = []
    plastic_predictions = []
    for i in range(len(model_predictions['boxes'])):
        if model_predictions['labels'][i] == 2:
            water_hyacinth_predictions.append(model_predictions['boxes'][i])
        elif model_predictions['labels'][i] == 1:
            plastic_predictions.append(model_predictions['boxes'][i])

    # find overlapping boudning boxes
    if len(plastic_predictions) ==0:
        return []
    else:
        overlapping_bboxes = []
        for (p_x1, p_y1, p_x2, p_y2) in plastic_predictions:
            plastic_top_left = {'x': p_x1.item(), 'y': p_y1.item()}
            plastic_btm_right = {'x': p_x2.item(), 'y':p_y2.item()}

            for (w_x1, w_y1, w_x2, w_y2) in water_hyacinth_predictions:
                hyacinth_top_left = {'x':w_x1.item(), 'y': w_y1.item()}
                hyacinth_btm_right = {'x':w_x2.item(), 'y': w_y2.item()}

                # if (plastic_top_left['x'] < hyacinth_btm_right['x'] or hyacinth_top_left['x'] < plastic_btm_right['x']):
                #     if (plastic_btm_right['y'] < hyacinth_top_left['y'] or hyacinth_btm_right['y'] < plastic_top_left['y']):
                if not (hyacinth_top_left['y'] >= plastic_btm_right['y'] or
                        hyacinth_btm_right['y'] <= plastic_top_left['y'] or 
                        hyacinth_top_left['x'] >= plastic_btm_right['x'] or 
                        hyacinth_btm_right['x'] <= plastic_top_left['x']):

                        trapped_plastic_bbox = [plastic_top_left['x'],
                                                    plastic_top_left['y'],
                                                    plastic_btm_right['x'],
                                                    plastic_btm_right['y']]
                        # don't add multiple times
                        if trapped_plastic_bbox not in overlapping_bboxes:
                            overlapping_bboxes.append(trapped_plastic_bbox)
        return overlapping_bboxes
    