import json
from dataclasses import dataclass
from utils.views import HorizontalFlipView, ViewTransform, Rotate180View
from typing import Optional, List
import random

@dataclass
class VideoPrompts:
    id: int
    prompt1: str
    prompt2: str
    views: List[ViewTransform]
    

class PromptGenerator:
    
    def __init__(self):
        with open("assets/visual_anagrams_prompt_bank.json", "r") as f:
            self.VisualAnagramPromptBank = json.load(f)
        
        with open("assets/heuristic_prompt_bank.json", "r") as f: 
            self.HeuristicPrompBank = json.load(f)
    
    def generate_static_visual_anagram_prompt(self, id: Optional[int], style: Optional[str], NumOfPrompts: int, preffered_view: Optional[str] = None) -> List[VideoPrompts]:
        """
        Generates a dual prompt for a static video that zooms into the optical illuison. You can use this for either HorizontalFlipView or Rotate180View.

        Args:
            id (Optional[int]): int from the prompt bank, or None to randomly select one
            NumOfPrompts (int): Number of randomly selected prompts to return if style is added it will include that style only. If view is added it will include that view only.
            
            style (Optional[str]): 'painting', 'street art', 'cubist painting', 'oil painting'
            
            preffered_view (Optional[str]): 'flip' or 'rotation'
            
        Returns:
            VideoPrompts: A dataclass containing the generated prompts and the view transformation.
        """
        #Validate inputs
        if preffered_view is not None and preffered_view not in ["flip", "rotation"]:
            raise ValueError(f"Invalid view: {preffered_view}. Must be 'flip' or 'rotation'.")
        
        if style is not None and style not in ["painting", "street art", "cubist painting", "oil painting"]:
            raise ValueError(f"Invalid style: {style}. Must be 'painting', 'street art', 'cubist painting', or 'oil painting'.")
        
        if NumOfPrompts is not None and (NumOfPrompts <= 0 or NumOfPrompts > len(self.VisualAnagramPromptBank)):
            raise ValueError(f"Invalid number of prompts: {NumOfPrompts}. Must be between 1 and {len(self.VisualAnagramPromptBank)}.")
        
        #Return one specific prompt with that ID
        if id is not None:
            promptData = next((prompt for prompt in self.VisualAnagramPromptBank if prompt["id"] == id), None)
            if promptData is None:
                raise ValueError(f"No prompt found with id {id}")
            if style is None:
                    style = promptData["style"].removesuffix(" of")
            prompt1 = f"A static video zooming into {promptData['style']} {promptData['subject1']}"
            prompt2 = f"A static video zooming into {promptData['style']} {promptData['subject2']}"
            views = []
            for view in promptData["transformations"]:
                if preffered_view == "flip" and view == "flip":
                    views.append(HorizontalFlipView())
                elif preffered_view == "rotation" and view == "rotation": 
                    views.append(Rotate180View())
                    
            video_prompts = VideoPrompts(
                id=promptData["id"],
                prompt1= prompt1,
                prompt2= prompt2,
                views= views)
            return [video_prompts]
        
        #Return a list of prompts based on the filters provided. If no filters randomly sample the prompts.
        elif NumOfPrompts is not None:
            filteredPrompts = self.VisualAnagramPromptBank
            if style is not None:
                style_match = "a " + style + " of" if style != "oil painting" else "an oil painting of"
                filteredPrompts = [prompt for prompt in self.VisualAnagramPromptBank if style_match in prompt["style"]]
            if preffered_view is not None:
                filteredPrompts = [prompt for prompt in filteredPrompts if preffered_view in prompt["transformations"]]
            
            if len(filteredPrompts) < NumOfPrompts:
                raise ValueError(f"Not enough prompts found with the given filters. Found {len(filteredPrompts)} prompts, but {NumOfPrompts} were requested.")
            
            sampledPrompts = random.sample(filteredPrompts, NumOfPrompts)
            prompts = []
            for promptData in sampledPrompts:
                if style is None:
                    style = promptData["style"].removesuffix(" of")
                prompt1 = f"Static video with no movement of {promptData['subject1']} in the style of {style}, high quality"
                prompt2 = f"Static video with no movement of {promptData['subject2']} in the style of {style}, high quality"
                if preffered_view is None:
                    views = []
                    for view in promptData["transformations"]:
                        if view == "flip":
                            views.append(HorizontalFlipView())
                        elif view == "rotate": 
                            views.append(Rotate180View())
                else:
                    views = [HorizontalFlipView()] if preffered_view == "flip" else [Rotate180View()]
                video_prompts = VideoPrompts(
                    id=promptData["id"],
                    prompt1= prompt1,
                    prompt2= prompt2,
                    views= views)
                prompts.append(video_prompts)
            return prompts
        
    def generate_static_heuristic_prompt(self, id: Optional[int], styles: Optional[List[str]], NumOfPrompts: int, preffered_view: Optional[str] = None) -> List[VideoPrompts]:
        """
        Generates a dual prompt for a static video that zooms into the optical illusion using the heuristic prompt bank.
        You can use this for either HorizontalFlipView or Rotate180View.

        Args:
            id (Optional[int]): int from the heuristic prompt bank, or None to randomly select one
            NumOfPrompts (int): Number of randomly selected prompts to return if style is added it will include that style only. If view is added it will include that view only.
            styles (Optional[List[str]]): List of styles to filter by. If None, all styles are included.
            preffered_view (Optional[str]): 'flip' or 'rotation'
        Returns:
            VideoPrompts: A dataclass containing the generated prompts and the view transformation.
        """
        # Validate inputs
        if preffered_view is not None and preffered_view not in ["flip", "rotation"]:
            raise ValueError(f"Invalid view: {preffered_view}. Must be 'flip' or 'rotation'.")
        
        valid_styles = ['street art', 'minimalist illustration', 'cubist painting', 'watercolor painting', 'vintage poster', 'pencil sketch', 'oil painting']
        if styles is not None:
            for s in styles:
                if s not in valid_styles:
                    raise ValueError(f"Invalid style: {s}. Must be one of {valid_styles}.")
        
        if NumOfPrompts is not None and (NumOfPrompts <= 0 or NumOfPrompts > len(self.HeuristicPrompBank)):
            raise ValueError(f"Invalid number of prompts: {NumOfPrompts}. Must be between 1 and {len(self.HeuristicPrompBank)}.")
        
        # Return one specific prompt with that ID
        if id is not None:
            promptData = next((prompt for prompt in self.HeuristicPrompBank if prompt["id"] == id), None)
            if promptData is None:
                raise ValueError(f"No prompt found with id {id}")
           
            style = promptData["style"].removesuffix(" of")
            prompt1 = f"Static video with no movement of {promptData['subject1']} in the style of {style}, high quality"
            prompt2 = f"Static video with no movement of {promptData['subject2']} in the style of {style}, high quality"
            views = []
            for view in promptData["transformations"]:
                if view == "flip":
                    views.append(HorizontalFlipView())
                elif view == "rotation": 
                    views.append(Rotate180View())
                    
            video_prompts = VideoPrompts(
                id=promptData["id"],
                prompt1=prompt1,
                prompt2=prompt2,
                views=views)
            return [video_prompts]
        
        # Return a list of prompts based on the filters provided. If no filters randomly sample the prompts.
        elif NumOfPrompts is not None:
            filteredPrompts = self.HeuristicPrompBank
            if styles is not None:
                # Build the set of style strings to match against
                style_matches = set()
                for s in styles:
                    style_match = "a " + s + " of" if s != "oil painting" else "an oil painting of"
                    style_matches.add(style_match)
                filteredPrompts = [prompt for prompt in self.HeuristicPrompBank if prompt["style"] in style_matches]
            
            if preffered_view is not None:
                filteredPrompts = [prompt for prompt in filteredPrompts if preffered_view in prompt["transformations"]]
            
            if len(filteredPrompts) < NumOfPrompts:
                raise ValueError(f"Not enough prompts found with the given filters. Found {len(filteredPrompts)} prompts, but {NumOfPrompts} were requested.")
            
            sampledPrompts = random.sample(filteredPrompts, NumOfPrompts)
            prompts = []
            for promptData in sampledPrompts:
                chosen_style = promptData["style"].removesuffix(" of")
                
                prompt1 = f"Static video with no movement of {promptData['subject1']} in the style of {chosen_style}, high quality"
                prompt2 = f"Static video with no movement of {promptData['subject2']} in the style of {chosen_style}, high quality"
                if preffered_view is None:
                    views = []
                    for view in promptData["transformations"]:
                        if view == "flip":
                            views.append(HorizontalFlipView())
                        elif view == "rotation":
                            views.append(Rotate180View())
                else:
                    views = [HorizontalFlipView()] if preffered_view == "flip" else [Rotate180View()]
                
                video_prompts = VideoPrompts(
                    id=promptData["id"],
                    prompt1=prompt1,
                    prompt2=prompt2,
                    views=views)
                prompts.append(video_prompts)
            return prompts