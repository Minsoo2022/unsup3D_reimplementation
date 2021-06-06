from networks import *
import torch
from torch import nn
import os
import math
import neural_renderer

class PhotoAE():
    def __init__(self, opt):
        self.device = 'cuda'
        self.image_size = opt.image_size
        self.Albedo_net = Encoder_Decoder(opt, opt.ch_img).to(self.device)
        self.Depth_net = Encoder_Decoder(opt, 1, value_norm=True).to(self.device)
        self.View_net = Encoder(opt, 6).to(self.device)
        self.Light_net = Encoder(opt, 4).to(self.device)
        self.Conf_net = Conf_Encoder_decoder(opt).to(self.device)

        self.max_depth = 1.1
        self.min_depth = 0.9

        self.renderer = Renderer(opt)
        self.percept_loss = PerceptualLoss().to(self.device)

        self.optim = torch.optim.Adam(list(self.Albedo_net.parameters()) + list(self.Depth_net.parameters())
                                      + list(self.View_net.parameters()) + list(self.Light_net.parameters())
                                      + list(self.Conf_net.parameters()),
                                      opt.lr, betas=[0.9,0.999]) #, weight_decay=5e-4
        self.L1_loss = nn.L1Loss()

    def save(self, opt, epoch):
        path = os.path.join(opt.checkpoint_dir, opt.name)
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'Albedo_net': self.Albedo_net.cpu().state_dict(),
            'Depth_net': self.Depth_net.cpu().state_dict(),
            'View_net': self.View_net.cpu().state_dict(),
            'Light_net': self.Light_net.cpu().state_dict(),
            'Conf_net': self.Conf_net.cpu().state_dict(),
            'optim': self.optim.state_dict(),
        }, os.path.join(path, f'model_{epoch}'))

        self.Albedo_net.to(self.device)
        self.Depth_net.to(self.device)
        self.View_net.to(self.device)
        self.Light_net.to(self.device)
        self.Conf_net.to(self.device)

    def load(self, opt):
        path = os.path.join(opt.load_dir)
        file = torch.load(path, map_location=self.device)
        self.Albedo_net.load_state_dict((file['Albedo_net']), strict=False)
        self.Depth_net.load_state_dict((file['Depth_net']), strict=False)
        self.View_net.load_state_dict((file['View_net']), strict=False)
        self.Light_net.load_state_dict((file['Light_net']), strict=False)
        self.Conf_net.load_state_dict((file['Conf_net']), strict=False)
        self.optim.load_state_dict((file['optim']))
        print('model load')

    def to_train(self):
        self.Albedo_net.train()
        self.Depth_net.train()
        self.View_net.train()
        self.Light_net.train()
        self.Conf_net.train()

    def to_eval(self):
        self.Albedo_net.eval()
        self.Depth_net.eval()
        self.View_net.eval()
        self.Light_net.eval()
        self.Conf_net.eval()

    def train(self, img):
        img = img.to(self.device)
        batch_size = img.shape[0]

        albedo = self.Albedo_net(img)
        albedo_prime = albedo.flip(3)

        depth = self.Depth_net(img).squeeze(1)

        border_zero = torch.ones(depth.shape)
        border_zero[:, :, :2] = 0
        border_zero[:, :, -2:] = 0
        border_one = 1 - border_zero
        border_zero = border_zero.to(self.device)
        border_one = border_one.to(self.device)
        depth = depth * border_zero + border_one

        depth = (1 + depth) * self.max_depth / 2 + (1 - depth) * self.min_depth / 2
        depth_prime = depth.flip(2)

        viewpoint = self.View_net(img)

        light = self.Light_net(img)
        ambience = light[:,:1] / 2 + 0.5
        diffuse = light[:, 1:2] / 2 + 0.5
        light_direction = light[:,2:]
        light_direction = torch.cat([light_direction, torch.ones(batch_size, 1).to(self.device)], 1)
        light_direction = light_direction / ((light_direction ** 2).sum(1, keepdim=True)) ** 0.5

        conf_, conf_percept_ = self.Conf_net(img)
        conf = conf_[:,0,:,:].unsqueeze(1)
        conf_prime = conf_[:,1,:,:].unsqueeze(1)
        conf_percept = conf_percept_[:, 0, :, :].unsqueeze(1)
        conf_percept_prime = conf_percept_[:, 1, :, :].unsqueeze(1)


        cannon_img, shading = self.renderer.render_cannonical(albedo, depth, ambience, diffuse, light_direction)
        cannon_img_prime, shading_prime = self.renderer.render_cannonical(albedo_prime, depth_prime, ambience, diffuse, light_direction)

        img_hat, recon_depth, _ = self.renderer.reprojection(cannon_img, depth, viewpoint)
        img_hat_prime, recon_depth_prime, _ = self.renderer.reprojection(cannon_img_prime, depth_prime, viewpoint)

        recon_im_mask = (recon_depth < self.max_depth + 0.1).float()
        recon_im_mask_prime = (recon_depth_prime < self.max_depth + 0.1).float()
        mask = (recon_im_mask * recon_im_mask_prime).unsqueeze(1).detach()
        img_hat = img_hat * mask
        img_hat_prime = img_hat_prime * mask


        results_images = {'albedo': albedo, 'albedo_prime': albedo_prime, 'depth': depth, 'depth_prime': depth_prime,
                   'conf': conf, 'conf_prime': conf_prime, 'cannon_img': cannon_img, 'cannon_img_prime': cannon_img_prime,
                   'img_hat': img_hat, 'img_hat_prime': img_hat_prime, 'gt_img': img, 'gt_img_fliped': img.flip(3)}
        restuls_scalars = {'viewpoint': viewpoint, 'light': light}

        l1_loss = L1loss_with_confidence(img, img_hat, conf, mask) + L1loss_with_confidence(img, img_hat_prime, conf_prime, mask) * 0.5
        percept_loss = self.percept_loss(img, img_hat, conf_percept, mask) + self.percept_loss(img, img_hat_prime, conf_percept_prime, mask) * 0.5

        total_loss = l1_loss + percept_loss
        self.optim.zero_grad()
        total_loss.backward()
        self.optim.step()
        losses = {'whole_loss': total_loss, 'l1_loss': l1_loss.item(), 'percept_loss': percept_loss.item()}
        return losses, results_images, restuls_scalars

    def evaluate(self, img, gt):
        img = img.to(self.device)
        gt = gt.squeeze(1).to(self.device)
        batch_size = img.shape[0]

        albedo = self.Albedo_net(img)
        albedo_prime = albedo.flip(3)

        depth = self.Depth_net(img).squeeze(1)

        border_zero = torch.ones(depth.shape)
        border_zero[:, :, :2] = 0
        border_zero[:, :, -2:] = 0
        border_one = 1 - border_zero
        border_zero = border_zero.to(self.device)
        border_one = border_one.to(self.device)
        depth = depth * border_zero + border_one

        depth = (1 + depth) * self.max_depth / 2 + (1 - depth) * self.min_depth / 2
        depth_prime = depth.flip(2)

        viewpoint = self.View_net(img)

        light = self.Light_net(img)
        ambience = light[:, :1] / 2 + 0.5
        diffuse = light[:, 1:2] / 2 + 0.5
        light_direction = light[:, 2:]
        light_direction = torch.cat([light_direction, torch.ones(batch_size, 1).to(self.device)], 1)
        light_direction = light_direction / ((light_direction ** 2).sum(1, keepdim=True)) ** 0.5

        cannon_img, shading = self.renderer.render_cannonical(albedo, depth, ambience, diffuse, light_direction)
        cannon_img_prime, shading_prime = self.renderer.render_cannonical(albedo_prime, depth_prime, ambience, diffuse,
                                                                          light_direction)

        img_hat, recon_depth, _ = self.renderer.reprojection(cannon_img, depth, viewpoint)
        img_hat_prime, recon_depth_prime, _ = self.renderer.reprojection(cannon_img_prime, depth_prime, viewpoint)

        recon_im_mask = (recon_depth < self.max_depth + 0.1).float().detach()
        recon_im_mask_prime = (recon_depth_prime < self.max_depth + 0.1).float()
        mask_pred = (nn.functional.avg_pool2d(recon_im_mask.unsqueeze(1), 3, stride=1, padding=1).squeeze(1) > 0.99).float()

        gt = (1 - gt) * 2 - 1
        gt = (1 + gt) * self.max_depth / 2 + (1 - gt) * self.min_depth / 2

        mask_gt = (gt < gt.max()).float()
        mask_gt = (nn.functional.avg_pool2d(mask_gt.unsqueeze(1), 3, stride=1, padding=1).squeeze(1) > 0.99).float()
        mask = mask_pred * mask_gt

        SIDE = cal_SIDE(recon_depth.log(), gt.log(), mask=mask)
        #self.acc_sie_masked = (self.sie_map_masked.view(batch_size,-1).sum(1) / mask.view(batch_size,-1).sum(1))**0.5

        recon_normal = self.renderer.cal_normal(recon_depth)
        gt_normal = self.renderer.cal_normal(gt)
        self.norm_err_map_masked = cal_MAD(recon_normal, gt_normal, mask=mask)
        self.acc_normal_masked = self.norm_err_map_masked.view(batch_size, -1).sum(1) / mask.view(batch_size, -1).sum(1)

        return SIDE, self.acc_normal_masked

    def visualize(self, img, gt=None):
        img = img.to(self.device)
        if gt is not None:
            gt = gt.squeeze(1).to(self.device)
        batch_size = img.shape[0]

        albedo = self.Albedo_net(img)
        albedo_prime = albedo.flip(3)

        depth = self.Depth_net(img).squeeze(1)

        depth = (1 + depth) * self.max_depth / 2 + (1 - depth) * self.min_depth / 2
        border_zero = torch.ones(depth.shape)
        border_zero[:, :, :2] = 0
        border_zero[:, :, -2:] = 0
        border_one = 1 - border_zero
        border_zero = border_zero.to(self.device)
        border_one = border_one.to(self.device)
        depth = depth * border_zero + border_one * (0.7*self.max_depth + 0.3*self.min_depth)


        depth_prime = depth.flip(2)

        viewpoint = self.View_net(img)

        light = self.Light_net(img)
        ambience = light[:,:1] / 2 + 0.5
        diffuse = light[:, 1:2] / 2 + 0.5
        light_direction = light[:,2:]
        light_direction = torch.cat([light_direction, torch.ones(batch_size, 1).to(self.device)], 1)
        light_direction = light_direction / ((light_direction ** 2).sum(1, keepdim=True)) ** 0.5

        conf_, conf_percept_ = self.Conf_net(img)
        conf = conf_[:,0,:,:].unsqueeze(1)
        conf_prime = conf_[:,1,:,:].unsqueeze(1)
        conf_percept = conf_percept_[:, 0, :, :].unsqueeze(1)
        conf_percept_prime = conf_percept_[:, 1, :, :].unsqueeze(1)


        cannon_img, shading = self.renderer.render_cannonical(albedo, depth, ambience, diffuse, light_direction)
        cannon_img_prime, shading_prime = self.renderer.render_cannonical(albedo_prime, depth_prime, ambience, diffuse, light_direction)

        img_hat, recon_depth, grid_2d_cannon = self.renderer.reprojection(cannon_img, depth, viewpoint)
        img_hat_prime, recon_depth_prime, _ = self.renderer.reprojection(cannon_img_prime, depth_prime, viewpoint)

        angle_list = [60, 30, 0, 0, -30, -60]
        recon_img_list = []
        recon_depth_list = []
        for i, angle in enumerate(angle_list):
            #view = torch.cuda.FloatTensor([0, angle * math.pi / 180, 0, 0, 0, -0.8713]).unsqueeze(0).repeat(batch_size, 1)
            view = torch.cuda.FloatTensor([0, angle * math.pi / 180, 0]).unsqueeze(0).repeat(batch_size, 1)
            view = torch.cat((view,viewpoint[:,3:]),dim=1)
            recon_img, _, _ = self.renderer.reprojection(cannon_img, depth, view)
            recon_shading, _, _= self.renderer.reprojection(shading, depth, view)
            if i < 3:
                recon_img_list.append((recon_shading*2 -2).repeat(1,3,1,1))
            else:
                recon_img_list.append(recon_img)
            recon_depth_list.append(recon_depth_list)
        recon_rotation_img = torch.stack(recon_img_list, dim=1)

        vertical_line = torch.zeros(self.image_size, self.image_size).to(self.device)
        vertical_line[:, self.image_size // 2 - 1:self.image_size // 2 + 1] = 1
        symmetric_line = nn.functional.grid_sample(vertical_line.repeat(batch_size, 1, 1, 1), grid_2d_cannon, mode='bilinear')
        img_with_sym_line = (0.5*symmetric_line) * torch.FloatTensor([-1,-1,1]).view(1,3,1,1).to(self.device) + (1-0.5*symmetric_line) * img

        conf_prime_vis = conf_prime.repeat(1,3,1,1) * torch.FloatTensor([1,-1,-1]).view(1,3,1,1).to(self.device) * 0.95 + (1-0.95*conf_prime) * img

        recon_im_mask = (recon_depth < self.max_depth + 0.1).float()
        recon_im_mask_prime = (recon_depth_prime < self.max_depth + 0.1).float()
        mask = (recon_im_mask * recon_im_mask_prime).unsqueeze(1).detach()
        img_hat = img_hat * mask
        img_hat_prime = img_hat_prime * mask

        mask_pred = (nn.functional.avg_pool2d(recon_im_mask.unsqueeze(1), 3, stride=1, padding=1).squeeze(
            1) > 0.99).float()
        if gt is not None:
            gt = (1 - gt) * 2 - 1
            gt = (1 + gt) * self.max_depth / 2 + (1 - gt) * self.min_depth / 2

            mask_gt = (gt < gt.max()).float()
            mask_gt = (nn.functional.avg_pool2d(mask_gt.unsqueeze(1), 3, stride=1, padding=1).squeeze(1) > 0.99).float()
            mask = mask_pred * mask_gt


        results_images = {'albedo': albedo, 'albedo_prime': albedo_prime, 'depth': depth, 'depth_prime': depth_prime,
                   'conf': conf, 'conf_prime': conf_prime, 'cannon_img': cannon_img, 'cannon_img_prime': cannon_img_prime,
                   'img_hat': img_hat, 'img_hat_prime': img_hat_prime, 'gt_img': img, 'gt_img_fliped': img.flip(3),
                          'recon_depth': recon_depth, 'recon_depth_masked': recon_depth * mask, 'recon_rotation_img' : recon_rotation_img,
                          'img_with_sym_line': img_with_sym_line, 'conf_prime_vis': conf_prime_vis}

        if gt is not None:
            results_images['gt'] = gt
            results_images['gt_masked'] = gt * mask

        restuls_scalars = {'viewpoint': viewpoint, 'light': light}

        return results_images


class Renderer():
    def __init__(self, opt):
        self.device = 'cuda'
        self.min_depth = 0.9
        self.max_depth = 1.1
        self.image_size = opt.image_size
        fx = (self.image_size - 1) / 2 / (math.tan(10 / 2 * math.pi / 180))
        fy = (self.image_size - 1) / 2 / (math.tan(10 / 2 * math.pi / 180))
        cx = (self.image_size - 1) / 2
        cy = (self.image_size - 1) / 2

        rotation = [[[1., 0., 0.],
              [0., 1., 0.],
              [0., 0., 1.]]]
        rotation = torch.FloatTensor(rotation).to(self.device)
        translation = torch.zeros(1, 3, dtype=torch.float32).to(self.device)

        K = [[fx, 0., cx],
             [0., fy, cy],
             [0., 0., 1.]]
        K = torch.FloatTensor(K).to(self.device)
        self.K_inverse = torch.inverse(K).unsqueeze(0)
        self.K = K.unsqueeze(0)

        self.renderer = neural_renderer.Renderer(
                                    light_intensity_ambient=1.0,
                                    light_intensity_directional=0.,
                                    K=self.K, R=rotation, t=translation,
                                    near=0.1, far=10,
                                    image_size=self.image_size, orig_size=self.image_size,
                                    background_color=[1,1,1]
                                    )

    def make_grid(self, batch_size, height, width):
        h_range = torch.arange(0, height)
        w_range = torch.arange(0, width)
        grid = torch.stack(torch.meshgrid([h_range, w_range]), -1).repeat(batch_size, 1, 1, 1).flip(3).float()
        return grid

    def cal_f(self, b, h, w):
        idx_map = torch.arange(h * w).reshape(h, w)
        faces1 = torch.stack([idx_map[:h - 1, :w - 1], idx_map[1:, :w - 1], idx_map[:h - 1, 1:]], -1).reshape(-1, 3)
        faces2 = torch.stack([idx_map[:h - 1, 1:], idx_map[1:, :w - 1], idx_map[1:, 1:]], -1).reshape(-1, 3)
        return torch.cat([faces1, faces2], 0).repeat(b, 1, 1).int()

    def cal_R(self, tx, ty, tz):
        n = len(tx)
        X = torch.zeros((n, 3, 3)).to(tx.device)
        Y = torch.zeros((n, 3, 3)).to(tx.device)
        Z = torch.zeros((n, 3, 3)).to(tx.device)

        X[:, 1, 1], X[:, 1, 2] = tx.cos(), -tx.sin()
        X[:, 2, 1], X[:, 2, 2] = tx.sin(), tx.cos()
        X[:, 0, 0] = 1

        Y[:, 0, 0], Y[:, 0, 2] = ty.cos(), ty.sin()
        Y[:, 2, 0], Y[:, 2, 2] = -ty.sin(), ty.cos()
        Y[:, 1, 1] = 1

        Z[:, 0, 0], Z[:, 0, 1] = tz.cos(), -tz.sin()
        Z[:, 1, 0], Z[:, 1, 1] = tz.sin(), tz.cos()
        Z[:, 2, 2] = 1
        return torch.matmul(Z, torch.matmul(Y, X))

    def cal_R_T(self, view):
        batch_size = view.size(0)
        if view.size(1) == 6:
            X = view[:, 0]
            Y = view[:, 1]
            Z = view[:, 2]
            translation_parm = view[:, 3:].reshape(batch_size, 1, 3)
        elif view.size(1) == 5:
            X = view[:, 0]
            Y = view[:, 1]
            Z = view[:, 2]
            differ = view[:, 3:].reshape(batch_size, 1, 2)
            translation_parm = torch.cat([differ, torch.zeros(batch_size, 1, 1).to(view.device)], 2)
        elif view.size(1) == 3:
            X = view[:, 0]
            Y = view[:, 1]
            Z = view[:, 2]
            translation_parm = torch.zeros(batch_size, 1, 3).to(view.device)
        matrix_R = self.cal_R(X, Y, Z)
        return matrix_R, translation_parm

    def rotation_function(self, points, matrix_R):
        mid = torch.FloatTensor([0., 0., (self.min_depth + self.max_depth) / 2]).to(self.device).view(1, 1, 3)
        points = points - mid
        points = points.matmul(matrix_R.transpose(2, 1))  # rotate
        points = points + mid
        return points

    def make_upgrid(self, depth):
        batch_size, height, width = depth.shape
        pixel_grid = self.make_grid(batch_size, height, width).to(self.device)
        depth = depth.unsqueeze(-1)
        up_grid = torch.cat((pixel_grid, torch.ones_like(depth)), dim=3)
        up_grid = up_grid.matmul(self.K_inverse.to(self.device).transpose(2, 1)) * depth
        return up_grid

    def projection_function(self, up_grid):
        batch_size, height, width, _ = up_grid.shape
        pixel_grid = up_grid / up_grid[..., 2:]
        pixel_grid = pixel_grid.matmul(self.K.to(self.device).transpose(2, 1))[:, :, :, :2]
        WH = torch.FloatTensor([width - 1, height - 1]).to(self.device).view(1, 1, 1, 2)
        pixel_grid = pixel_grid / WH * 2. - 1.  # normalize to -1~1
        return pixel_grid

    def cal_upgrid(self, depth):
        b, h, w = depth.shape
        up_grid = self.make_upgrid(depth).reshape(b, -1, 3)
        up_grid = self.rotation_function(up_grid, self.matrix_R)
        up_grid = up_grid + self.translation_parm
        return up_grid.reshape(b, h, w, 3)  # return 3d vertices

    def cal_inverse_upgrid(self, depth):
        batch_size, height, width = depth.shape
        up_grid = self.make_upgrid(depth).reshape(batch_size, -1, 3)
        up_grid = up_grid - self.translation_parm
        up_grid = self.rotation_function(up_grid, self.matrix_R.transpose(2, 1))
        return up_grid.reshape(batch_size, height, width, 3)  # return 3d vertices

    def get_inv_warped_2d_grid(self, depth):
        up_grid = self.cal_inverse_upgrid(depth)
        pixel_grid = self.projection_function(up_grid)
        return pixel_grid

    def cal_recon_depth(self, canon_depth):
        batch_size, height, width = canon_depth.shape
        up_grid = self.cal_upgrid(canon_depth).reshape(batch_size, -1, 3)
        faces = self.cal_f(batch_size, height, width).to(self.device)
        randered_depth = self.renderer.render_depth(up_grid, faces)
        randered_depth = randered_depth.clamp(min=self.min_depth - (self.max_depth - self.min_depth) / 2, max=self.max_depth + (self.max_depth - self.min_depth) / 2)
        return randered_depth

    def cal_normal(self, depth):
        batch_size, height, width = depth.shape
        up_grid = self.make_upgrid(depth)

        tu = up_grid[:, 1:-1, 2:] - up_grid[:, 1:-1, :-2]
        tv = up_grid[:, 2:, 1:-1] - up_grid[:, :-2, 1:-1]
        normal = tu.cross(tv, dim=3)

        zero = torch.FloatTensor([0, 0, 1]).to(self.device)
        normal = torch.cat([zero.repeat(batch_size, height - 2, 1, 1), normal, zero.repeat(batch_size, height - 2, 1, 1)], 2)
        normal = torch.cat([zero.repeat(batch_size, 1, width, 1), normal, zero.repeat(batch_size, 1, width, 1)], 1)
        normal = normal / (((normal ** 2).sum(3, keepdim=True)) ** 0.5 + 1e-7)
        return normal

    def render_cannonical(self, albedo, depth, ambience, diffuse, light_direction):
        normal = self.cal_normal(depth)
        shading = torch.sum((normal * light_direction.view(-1, 1, 1, 3)), dim=3).clamp(min=0).unsqueeze(1) * \
                  ambience.view(-1, 1, 1, 1) + diffuse.view(-1, 1, 1, 1)
        cannonical = ((albedo / 2 + 0.5) * shading - 0.5) * 2

        return cannonical, shading

    def reprojection(self, cannon_img, depth, viewpoint):
        viewpoint = torch.cat([
            viewpoint[:, :3] * math.pi / 180 * 60,
            viewpoint[:, 3:5] * 0.1,
            viewpoint[:, 5:] * 0.1], 1)

        self.matrix_R, self.translation_parm = self.cal_R_T(viewpoint)
        recon_depth = self.cal_recon_depth(depth)
        recon_normal = self.cal_normal(recon_depth)
        pixel_grid_from_canon = self.get_inv_warped_2d_grid(recon_depth)
        recon_img = nn.functional.grid_sample(cannon_img, pixel_grid_from_canon, mode='bilinear')

        return recon_img, recon_depth, pixel_grid_from_canon
