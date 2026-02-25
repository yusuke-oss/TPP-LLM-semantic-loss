"""
Utility Functions
"""


def get_prompt(
    dataset_name: str, 
    event_time_first: bool,
) -> str:
    """
    Get the prompt for the TPP-LLM

    :param dataset_name: dataset name
    :param event_time_first: event time first (before the event type) for each event
    :return: prompt
    """
    sequence_descriptions = {
        "us_earthquake": "You are given a sequence of earthquake events recorded in the United States.",
        
    }

    event_descriptions = {
        
        "us_earthquake": {
            "event_type_first": "Each event in the sequence lists the magnitude classification (large or small) followed by the magnitude, followed by the depth, followed by the timestamp. <|start_of_event|> represents the start of an event and <|end_of_event|> represents the end of an event, <|time_prefix|> is Prefix token for event timestamp, <|type_prefix|> is Prefix token for event type, <|magnitude_prefix|> is Prefix token for event magnitude, <|depth_prefix|> is Prefix token for event depth.",
            "event_time_first": "Each event in the sequence lists the timestamp followed by the magnitude classification (large or small), followed by the magnitude, followed by the depth. <|start_of_event|> represents the start of an event and <|end_of_event|> represents the end of an event, <|time_prefix|> is Prefix token for event timestamp, <|type_prefix|> is Prefix token for event type, <|magnitude_prefix|> is Prefix token for event magnitude, <|depth_prefix|> is Prefix token for event depth."
        },
        
        
    }

    task_descriptions = {
        "event_type_first": "Based on this sequence, predict the next event type and the corresponding time.",
        "event_time_first": "Based on this sequence, predict the next event time and the corresponding type."
    }
    
    
    
    
    if dataset_name not in sequence_descriptions:
        return "Dataset not recognized."

    sequence_description = sequence_descriptions[dataset_name]
    if event_time_first:
        event_description = event_descriptions[dataset_name]["event_time_first"]
        task_description = task_descriptions["event_time_first"]
    else:
        event_description = event_descriptions[dataset_name]["event_type_first"]
        task_description = task_descriptions["event_type_first"]

    
    

    return f"{sequence_description} {event_description} {task_description} "
