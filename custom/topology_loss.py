import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as ssp
import ot
import math

class PersistenceImageTorch(torch.nn.Module):
    def __init__(self, bandwidth=0.01, weight= None, resolution=[20,20], im_range=torch.tensor([float('nan')] * 4), device='cuda'):
        super(PersistenceImageTorch, self).__init__()
        self.bandwidth = bandwidth
        self.weight = weight if weight is not None else None
        self.weight = weight
        self.resolution, self.im_range = resolution, torch.tensor(im_range).to(device)
        self.device = device
        self.to(device)
        self.bandwidth_sq = self.bandwidth ** 2
        self.norm_const = 2 * math.pi * self.bandwidth_sq


        self.im_range_fixed_ = self.im_range
        self.no_im_range_fixed_ = torch.isnan(self.im_range.detach()).any()

        self.x_values, self.y_values = torch.linspace(self.im_range_fixed_[0], self.im_range_fixed_[1], self.resolution[0], device=self.device), torch.linspace(self.im_range_fixed_[2], self.im_range_fixed_[3], self.resolution[1], device=self.device)

    def forward(self, X):
        # Parameters: X (list of n x 2 tensors) list of PDs, 
        num_diag, Xfit = len(X), []
        new_X = []
        for i in range(num_diag):
            dg = X[i].to(self.device)
            new_X += [torch.vstack((dg[:,0],dg[:,1] - dg[:,0])).T]      # transform to (birth, death-birth)

        # setting "im_range_fixed_"
        if self.no_im_range_fixed_:
            persistence_pts = torch.cat(new_X, dim=0)
            mx, Mx, my, My = persistence_pts[:,0].min(), persistence_pts[:,0].max(), persistence_pts[:,1].min(), persistence_pts[:,1].max()
            self.im_range_fixed_ = torch.where(
                torch.isnan(self.im_range.clone().detach()),
                torch.tensor([mx, Mx, my, My]).to(self.device),  
                self.im_range.clone().detach()
                )

        for i in range(num_diag):
            diagram, num_pts_in_diag = new_X[i], new_X[i].shape[0]
            if self.weight is not None:
                w = self.weight(diagram.t()) #### Fast

            Xs, Ys = torch.tile((diagram[:,0][:,None,None]-self.x_values[None,None,:]),[self.resolution[1],1]), torch.tile(diagram[:,1][:,None,None]-self.y_values[None,:,None],[1,1,self.resolution[0]])

            sq_dist = Xs.pow(2) + Ys.pow(2)  # [n, res_x, res_y]
            gauss = torch.exp(-sq_dist / (2 * self.bandwidth_sq)) / self.norm_const  # [n, res_x, res_y]
            if self.weight is None:
                weighted_gauss = gauss
            else:
                weighted_gauss = (w[:, None, None] * gauss)  # [n, res_x, res_y]
            image = weighted_gauss.sum(dim=0)  # [res_x, res_y]


            Xfit.append(image.flatten()[None,:])
        
        Xfit = torch.cat(Xfit, 0)

        return Xfit


class TopologyLossCalculator:
    def __init__(self, output_device="cpu", loss='swd', swd_k=50, loss_scale=1., std_scale=0.5, metric='euclidean'):
        self.device = output_device
        self.loss_scale = loss_scale
        self.swd_k = swd_k
        self.std_scale = std_scale
        self.metric = metric
        self.paring_loss_fn = nn.MSELoss(reduction='mean')
        self.PI_calculator = PersistenceImageTorch(resolution=[1,64], im_range=[0,0,0,1], device=output_device)
        self.pi_loss_fn = torch.nn.MSELoss()

    def __call__(self, points):
        dists = self.compute_distmat(points, std_scale=self.std_scale)
        dists_np = dists.detach().float().cpu().numpy()
        csr_dists = ssp.csr_matrix(dists_np)
        csr_upper = ssp.triu(csr_dists, k=1)  # k=1 -> strictly upper triangle

        # Compute MST (0-dimensional info)
        mst = ssp.csgraph.minimum_spanning_tree(csr_upper)  
        zerodim_info = set(zip(*mst.nonzero()))  # MST edges

        # 1-dim: edges not in MST
        dists_nonzero = set(zip(*csr_upper.nonzero()))
        onedim_info = dists_nonzero - zerodim_info

        # PD0 (connected components): birth=0, death=distance
        pd0 = torch.zeros((len(zerodim_info), 2), device=self.device)
        for k, (i, j) in enumerate(zerodim_info):
            pd0[k, 1] = dists[i, j]  # birth=0 (default), death=dists[i,j]

        # PD1 (loops): birth=distance, death=inf
        pd1 = torch.zeros((len(onedim_info), 2), device=self.device)
        for k, (i, j) in enumerate(onedim_info):
            pd1[k, 0] = dists[i, j]  # birth=dists[i,j]
            # pd1[k, 1] = 0  # death=inf

        return [pd0, pd1], zerodim_info, onedim_info

    def compute_distmat(self, points, std_scale=0.5):
        # points: (n_points, dim)
        if self.metric == 'euclidean':
            dists = torch.cdist(points, points, p=2)
        elif self.metric == 'similarlity':
            dists = 1 - points @ points.T

        if std_scale is not None:
            # Compute threshold from mean - std * scale
            mean = dists.mean()
            std = dists.std()
            threshold = mean - std_scale * std

            # Thresholding: disconnect edges larger than threshold
            mask = dists > threshold
            dists = dists.masked_fill(mask, 0)

        max_val = dists.max()
        dists = dists / (max_val+1e-8)

        return dists

    def swd_loss(self, output_pds, target_pds):
        # 0-dim
        if output_pds[0].shape[0] == 0:
            output_pds[0] = torch.zeros((1,2)).to(self.device)
        if target_pds[0].shape[0] == 0:
            target_pds[0] = torch.zeros((1,2)).to(self.device)
        zerodim_loss = ot.sliced_wasserstein_distance(output_pds[0], target_pds[0], n_projections=self.swd_k)

        # 1-dim
        if output_pds[1].shape[0] == 0:
            output_pds[1] = torch.zeros((1,2)).to(self.device)
        if target_pds[1].shape[0] == 0:
            target_pds[1] = torch.zeros((1,2)).to(self.device)
        onedim_loss = ot.sliced_wasserstein_distance(output_pds[1], target_pds[1], n_projections=self.swd_k)

        return self.loss_scale*zerodim_loss, self.loss_scale*onedim_loss

    def pi_loss(self, output_pds, target_pds):
        pis = self.PI_calculator([
            output_pds[0],
            target_pds[0],
            output_pds[1][:, [1, 0]],
            target_pds[1][:, [1, 0]]
        ])

        # Group-wise max for normalization
        max_0d = torch.max(pis[0].max(), pis[1].max()) + 1e-8
        max_1d = torch.max(pis[2].max(), pis[3].max()) + 1e-8

        pis = [
            pis[0] / max_0d,
            pis[1] / max_0d,
            pis[2] / max_1d,
            pis[3] / max_1d,
        ]

        zerodim_loss = self.pi_loss_fn(pis[0], pis[1])
        onedim_loss = self.pi_loss_fn(pis[2], pis[3])
        return self.loss_scale*zerodim_loss, self.loss_scale*onedim_loss
    
    def persistence_pairing_loss(self, persistence_pairing1, persistence_pairing2, points1, points2):
        loss = torch.zeros((), device=self.device)
        dists1 = self.compute_distmat(points1, std_scale=None)
        dists2 = self.compute_distmat(points2, std_scale=None)
        if len(dists1) == 0 or len(dists2) == 0:
            return loss

        if not len(list(persistence_pairing1)) ==0:
            i_idx, j_idx = zip(*list(persistence_pairing1))
            i_idx = torch.tensor(i_idx)
            j_idx = torch.tensor(j_idx)
            values1 = dists1[i_idx, j_idx]
            values2 = dists2[i_idx, j_idx]
            loss += (1/2) * self.paring_loss_fn(values1, values2)

        if not len(list(persistence_pairing2)) ==0:
            i_idx, j_idx = zip(*list(persistence_pairing2))
            i_idx = torch.tensor(i_idx)
            j_idx = torch.tensor(j_idx)
            values1 = dists1[i_idx, j_idx]
            values2 = dists2[i_idx, j_idx]
            loss += (1/2) * self.paring_loss_fn(values1, values2)

        return self.loss_scale*loss
    
    def persistence_pairing_similarlity_loss(self, persistence_pairing1, persistence_pairing2, points1, points2):
        loss = torch.zeros((), device=self.device)

        pairs1 = list(persistence_pairing1)
        pairs2 = list(persistence_pairing2)
        if len(pairs1) == 0 or len(pairs2) == 0:
            return loss

        if not len(list(persistence_pairing1)) ==0:
            vec1 = points1[torch.tensor(list(persistence_pairing1))[:,0]] - points1[torch.tensor(list(persistence_pairing1))[:,1]]
            vec2 = points2[torch.tensor(list(persistence_pairing1))[:,0]] - points2[torch.tensor(list(persistence_pairing1))[:,1]]
            vec1 = vec1 / (torch.norm(vec1, dim=1, keepdim=True) +1e-8)
            vec2 = vec2 / (torch.norm(vec2, dim=1, keepdim=True) +1e-8)
            loss += (1/2) * (1 - (vec1*vec2).sum(dim=-1).abs().mean(dim=0))
            # loss += (1/2) * (1 - (vec1*vec2).sum(dim=-1).mean(dim=0))

        if not len(list(persistence_pairing2)) ==0:
            vec1 = points1[torch.tensor(list(persistence_pairing2))[:,0]] - points1[torch.tensor(list(persistence_pairing2))[:,1]]
            vec2 = points2[torch.tensor(list(persistence_pairing2))[:,0]] - points2[torch.tensor(list(persistence_pairing2))[:,1]]
            vec1 = vec1 / (torch.norm(vec1, dim=1, keepdim=True) +1e-8)
            vec2 = vec2 / (torch.norm(vec2, dim=1, keepdim=True) +1e-8)
            loss += (1/2) * (1 - (vec1*vec2).sum(dim=-1).abs().mean(dim=0))
            # loss += (1/2) * (1 - (vec1*vec2).sum(dim=-1).mean(dim=0))

        return self.loss_scale*loss

    def filtered_persistence_pairing_similarlity_loss(self, persistence_pairing1, persistence_pairing2, points1, points2):
        loss = torch.zeros((), device=self.device)

        filtered_pairings = persistence_pairing1 and persistence_pairing2
        pairs = list(filtered_pairings)

        if len(pairs) == 0:
            return loss
        else:
            vec1 = points1[torch.tensor(list(filtered_pairings))[:,0]] - points1[torch.tensor(list(filtered_pairings))[:,1]]
            vec2 = points2[torch.tensor(list(filtered_pairings))[:,0]] - points2[torch.tensor(list(filtered_pairings))[:,1]]
            vec1 = vec1 / (torch.norm(vec1, dim=1, keepdim=True) +1e-8)
            vec2 = vec2 / (torch.norm(vec2, dim=1, keepdim=True) +1e-8)
            loss += 1 - (vec1*vec2).sum(dim=-1).abs().mean(dim=0)

            return self.loss_scale*loss
