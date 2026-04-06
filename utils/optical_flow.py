from os import read
from types import SimpleNamespace
import torch
from torchvision.io import read_video
from RAFT.core.raft import RAFT
from RAFT.core.utils.utils import InputPadder



class OpticalFlow:
    def _load_model(self):
        args = SimpleNamespace(
            small=False,
            mixed_precision=True,
            alternate_corr=False,
            dropout=0,
            )
        model = torch.nn.DataParallel(RAFT(args))
        model.load_state_dict(torch.load('RAFT/models/raft-sintel.pth'))
        model = model.module
        model.cuda()
        model.eval()
        
        return model
    
    def _extract_motion_from_video(self,model, frames_list):
        num_frames = len(frames_list)
        all_flows = []
        c,h,w = frames_list[0].shape
        with torch.no_grad():
            for i in range(num_frames - 1):
                image0 = frames_list[i].unsqueeze(0).cuda()
                image1 = frames_list[i + 1].unsqueeze(0).cuda()

                padder = InputPadder(image0.shape)
                image0, image1 = padder.pad(image0, image1)
                _, flow_up = model(image0, image1, iters=12, test_mode=True)
                flow_output = padder.unpad(flow_up).detach().cpu().squeeze().permute(1, 2, 0)
                all_flows.append(flow_output)
        return all_flows
    
    def generate_motion_tensors_from_video(self, video_path):
        frames = (read_video(video_path)[0]).float()
        frame_list = list(frames.permute(0,3,1,2))
        model = self._load_model()
        motion = self._extract_motion_from_video(model, frame_list)
        
        return motion

    