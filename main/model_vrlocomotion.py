import torch
import torch.nn as nn
import torch.nn.functional as F



class YNetEncoder(nn.Module):
	def __init__(self, in_channels, channels=(64, 128, 256, 512, 512)):
		"""
		Encoder model
		:param in_channels: int, semantic_classes + obs_len
		:param channels: list, hidden layer channels
		"""
		super(YNetEncoder, self).__init__()
		self.stages = nn.ModuleList()

		# First block
		self.stages.append(nn.Sequential(
			nn.Conv2d(in_channels, channels[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			nn.ReLU(inplace=True),
		))

		# Subsequent blocks, each starting with MaxPool
		for i in range(len(channels)-1):
			self.stages.append(nn.Sequential(
				nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
				nn.Conv2d(channels[i], channels[i+1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
				nn.ReLU(inplace=True),
				nn.Conv2d(channels[i+1], channels[i+1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
				nn.ReLU(inplace=True)))

		# Last MaxPool layer before passing the features into decoder
		self.stages.append(nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)))

	def forward(self, x):
		# Saves the feature maps Tensor of each layer into a list, as we will later need them again for the decoder
		features = []
		for stage in self.stages:
			x = stage(x)
			features.append(x)
		return features


class YNetDecoder(nn.Module):
	def __init__(self, encoder_channels, decoder_channels, output_len, traj=False):
		"""
		Decoder models
		:param encoder_channels: list, encoder channels, used for skip connections
		:param decoder_channels: list, decoder channels
		:param output_len: int, pred_len
		:param traj: False or int, if False -> Goal and waypoint predictor, if int -> number of waypoints
		"""
		super(YNetDecoder, self).__init__()

		# The trajectory decoder takes in addition the conditioned goal and waypoints as an additional image channel
		if traj:
			encoder_channels = [channel+traj for channel in encoder_channels]
		encoder_channels = encoder_channels[::-1]  # reverse channels to start from head of encoder
		center_channels = encoder_channels[0]

		decoder_channels = decoder_channels

		# The center layer (the layer with the smallest feature map size)
		self.center = nn.Sequential(
			nn.Conv2d(center_channels, center_channels*2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			nn.ReLU(inplace=True),
			nn.Conv2d(center_channels*2, center_channels*2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			nn.ReLU(inplace=True)
		)

		# Determine the upsample channel dimensions
		upsample_channels_in = [center_channels*2] + decoder_channels[:-1]
		upsample_channels_out = [num_channel // 2 for num_channel in upsample_channels_in]

		# Upsampling consists of bilinear upsampling + 3x3 Conv, here the 3x3 Conv is defined
		self.upsample_conv = [
			nn.Conv2d(in_channels_, out_channels_, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
			for in_channels_, out_channels_ in zip(upsample_channels_in, upsample_channels_out)]
		self.upsample_conv = nn.ModuleList(self.upsample_conv)

		# Determine the input and output channel dimensions of each layer in the decoder
		# As we concat the encoded feature and decoded features we have to sum both dims
		in_channels = [enc + dec for enc, dec in zip(encoder_channels, upsample_channels_out)]
		out_channels = decoder_channels

		self.decoder = [nn.Sequential(
			nn.Conv2d(in_channels_, out_channels_, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_channels_, out_channels_, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			nn.ReLU(inplace=True))
			for in_channels_, out_channels_ in zip(in_channels, out_channels)]
		self.decoder = nn.ModuleList(self.decoder)


		# Final 1x1 Conv prediction to get our heatmap logits (before softmax)
		self.predictor = nn.Conv2d(in_channels=decoder_channels[-1], out_channels=output_len, kernel_size=1, stride=1, padding=0)

	def forward(self, features):
		# Takes in the list of feature maps from the encoder. Trajectory predictor in addition the goal and waypoint heatmaps
		features = features[::-1]  # reverse the order of encoded features, as the decoder starts from the smallest image
		center_feature = features[0]
		x = self.center(center_feature)
		t = x.dtype
		for i, (feature, module, upsample_conv) in enumerate(zip(features[1:], self.decoder, self.upsample_conv)):
			x = F.interpolate(x.float(), scale_factor=2, mode='bilinear', align_corners=False).to(t)  # bilinear interpolation for upsampling
			x = upsample_conv(x)  # 3x3 conv for upsampling
			x = torch.cat([x, feature], dim=1)  # concat encoder and decoder features
			x = module(x)  # Conv
		x = self.predictor(x)  # last predictor layer
		return x

class PRED_GOAL(nn.Module):
	def __init__(self, obs_len,	pred_len, map_channel, encoder_channels=[], decoder_channels=[]):
		"""
		Complete Y-net Architecture including semantic segmentation backbone, heatmap embedding and ConvPredictor
		:param obs_len: int, observed timesteps
		:param pred_len: int, predicted timesteps
		:param segmentation_model_fp: str, filepath to pretrained segmentation model
		:param use_features_only: bool, if True -> use segmentation features from penultimate layer, if False -> use softmax class predictions
		:param semantic_classes: int, number of semantic classes
		:param encoder_channels: list, encoder channel structure
		:param decoder_channels: list, decoder channel structure
		:param waypoints: int, number of waypoints
		"""
		super(PRED_GOAL, self).__init__()

        #goal=1 + past_traj=15
		self.encoder = YNetEncoder(in_channels= map_channel + 35, channels=encoder_channels)
		self.decoder = YNetDecoder(encoder_channels, decoder_channels, output_len=1)

	def dec(self, features):
		v = self.decoder(features)
		return v

	def enc(self, x):
		features = self.encoder(x)
		return features

	def forward(self,x):
        
		f = self.enc(x)
		v = self.dec(f)
        
		return v

class PRED_GOAL2(nn.Module):
	def __init__(self, obs_len,	pred_len, map_channel, encoder_channels=[], decoder_channels=[]):
		"""
		Complete Y-net Architecture including semantic segmentation backbone, heatmap embedding and ConvPredictor
		:param obs_len: int, observed timesteps
		:param pred_len: int, predicted timesteps
		:param segmentation_model_fp: str, filepath to pretrained segmentation model
		:param use_features_only: bool, if True -> use segmentation features from penultimate layer, if False -> use softmax class predictions
		:param semantic_classes: int, number of semantic classes
		:param encoder_channels: list, encoder channel structure
		:param decoder_channels: list, decoder channel structure
		:param waypoints: int, number of waypoints
		"""
		super(PRED_GOAL2, self).__init__()

        #goal=1 + past_traj=15
		self.encoder = YNetEncoder(in_channels= map_channel + 35*2, channels=encoder_channels)
		self.decoder = YNetDecoder(encoder_channels, decoder_channels, output_len=1)

	def dec(self, features):
		v = self.decoder(features)
		return v

	def enc(self, x):
		features = self.encoder(x)
		return features

	def forward(self,x):
        
		f = self.enc(x)
		v = self.dec(f)
        
		return v

class PRED_GM(nn.Module):
	def __init__(self, obs_len,	pred_len, map_channel, encoder_channels=[], decoder_channels=[]):
		"""
		Complete Y-net Architecture including semantic segmentation backbone, heatmap embedding and ConvPredictor
		:param obs_len: int, observed timesteps
		:param pred_len: int, predicted timesteps
		:param segmentation_model_fp: str, filepath to pretrained segmentation model
		:param use_features_only: bool, if True -> use segmentation features from penultimate layer, if False -> use softmax class predictions
		:param semantic_classes: int, number of semantic classes
		:param encoder_channels: list, encoder channel structure
		:param decoder_channels: list, decoder channel structure
		:param waypoints: int, number of waypoints
		"""
		super(PRED_GM, self).__init__()

        #goal=1 + past_traj=15*2
		self.encoder = YNetEncoder(in_channels = map_channel + 1 + obs_len*2, channels=encoder_channels)
		self.decoder = YNetDecoder(encoder_channels, decoder_channels, output_len=2)

	def dec(self, features):
		v = self.decoder(features)
		return v

	def enc(self, x):
		features = self.encoder(x)
		return features
    
	def forward(self,x):
        
		f = self.enc(x)
		v = self.dec(f)
        
		return v

class PRED_TRAJ_ALONG_PATH(nn.Module):
	def __init__(self, obs_len,	pred_len, map_channel, encoder_channels=[], decoder_channels=[]):
		"""
		Complete Y-net Architecture including semantic segmentation backbone, heatmap embedding and ConvPredictor
		:param obs_len: int, observed timesteps
		:param pred_len: int, predicted timesteps
		:param segmentation_model_fp: str, filepath to pretrained segmentation model
		:param use_features_only: bool, if True -> use segmentation features from penultimate layer, if False -> use softmax class predictions
		:param semantic_classes: int, number of semantic classes
		:param encoder_channels: list, encoder channel structure
		:param decoder_channels: list, decoder channel structure
		:param waypoints: int, number of waypoints
		"""
		super(PRED_TRAJ_ALONG_PATH, self).__init__()

        #goal=1 + past_traj=15
		self.encoder = YNetEncoder(in_channels= map_channel + 1 + obs_len*2 + 1, channels=encoder_channels)
		self.decoder = YNetDecoder(encoder_channels, decoder_channels, output_len=pred_len)

	def dec(self, features):
		v = self.decoder(features)
		return v

	def enc(self, x):
		features = self.encoder(x)
		return features

	def forward(self,x):
        
		f = self.enc(x)
		v = self.dec(f)
        
		return v

class PRED_TRAJ_POS(nn.Module):
	def __init__(self, obs_len,	pred_len, map_channel, encoder_channels=[], decoder_channels=[]):
		"""
		Complete Y-net Architecture including semantic segmentation backbone, heatmap embedding and ConvPredictor
		:param obs_len: int, observed timesteps
		:param pred_len: int, predicted timesteps
		:param segmentation_model_fp: str, filepath to pretrained segmentation model
		:param use_features_only: bool, if True -> use segmentation features from penultimate layer, if False -> use softmax class predictions
		:param semantic_classes: int, number of semantic classes
		:param encoder_channels: list, encoder channel structure
		:param decoder_channels: list, decoder channel structure
		:param waypoints: int, number of waypoints
		"""
		super(PRED_TRAJ_POS, self).__init__()

        #goal=1 + past_traj=15
		self.encoder = YNetEncoder(in_channels= 2, channels=encoder_channels)
		self.decoder = YNetDecoder(encoder_channels, decoder_channels, output_len=pred_len)

		self.fc = nn.Sequential(
            nn.Linear(in_features=160, out_features=256),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=256, out_features=256),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=256, out_features=pred_len*2),
        )
        
	def dec(self, f1, f2):
		v1 = self.fc1(torch.flatten(f1[5],start_dim=1))
		v2 = self.fc2(torch.flatten(f2,start_dim=1))
        
		v = self.fc3(torch.cat([v1,v2],dim=1))
        
		return v

	def enc(self, x):
		features = self.encoder(x)
		return features

	def forward(self,x):
        
		f = self.fc(torch.flatten(x,start_dim=1))
        
		return f