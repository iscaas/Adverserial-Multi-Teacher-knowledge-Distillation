import torch
import torch.nn as nn
import tensorflow as tf
from ..attack import Attack


class VNIFGSM(Attack):
    r"""
    VNI-FGSM in the paper 'Enhancing the Transferability of Adversarial Attacks through Variance Tuning
    [https://arxiv.org/abs/2103.15571], Published as a conference paper at CVPR 2021
    Modified from "https://github.com/JHL-HUST/VT"

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of iterations. (Default: 10)
        decay (float): momentum factor. (Default: 1.0)
        N (int): the number of sampled examples in the neighborhood. (Default: 5)
        beta (float): the upper bound of neighborhood. (Default: 3/2)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.VNIFGSM(model, eps=8/255, alpha=2/255, steps=10, decay=1.0, N=5, beta=3/2)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, device=None, eps=8/255, alpha=2/255, steps=10, decay=1.0, N=5, beta=3/2):
        super().__init__('VNIFGSM', model, device)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.N = N
        self.beta = beta
        self.supported_mode = ['default', 'targeted']

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        # images = images.clone().detach().to(self.device)
        # labels = labels.clone().detach().to(self.device)

        # if self.targeted:
        #     target_labels = self.get_target_label(images, labels)

        # momentum = torch.zeros_like(images).detach().to(self.device)
        # v = torch.zeros_like(images).detach().to(self.device)
        # loss = nn.CrossEntropyLoss()
        # adv_images = images.clone().detach()

        # for _ in range(self.steps):
        #     adv_images.requires_grad = True
        #     nes_images = adv_images + self.decay * self.alpha * momentum
        #     outputs = self.get_logits(nes_images)

        #     # Calculate loss
        #     if self.targeted:
        #         cost = -loss(outputs, target_labels)
        #     else:
        #         cost = loss(outputs, labels)

        #     # Update adversarial images
        #     adv_grad = torch.autograd.grad(cost, adv_images,
        #                                    retain_graph=False, create_graph=False)[0]

        #     grad = (adv_grad + v) / torch.mean(torch.abs(adv_grad + v),
        #                                        dim=(1, 2, 3), keepdim=True)
        #     grad = grad + momentum * self.decay
        #     momentum = grad

        #     # Calculate Gradient Variance
        #     GV_grad = torch.zeros_like(images).detach().to(self.device)
        #     for _ in range(self.N):
        #         neighbor_images = adv_images.detach() + \
        #             torch.randn_like(images).uniform_(-self.eps *
        #                                               self.beta, self.eps*self.beta)
        #         neighbor_images.requires_grad = True
        #         outputs = self.get_logits(neighbor_images)

        #         # Calculate loss
        #         if self.targeted:
        #             cost = -loss(outputs, target_labels)
        #         else:
        #             cost = loss(outputs, labels)
        #         GV_grad += torch.autograd.grad(cost, neighbor_images,
        #                                        retain_graph=False, create_graph=False)[0]
        #     # obtaining the gradient variance
        #     v = GV_grad / self.N - adv_grad

        #     adv_images = adv_images.detach() + self.alpha*grad.sign()
        #     delta = torch.clamp(adv_images - images,
        #                         min=-self.eps, max=self.eps)
        #     adv_images = torch.clamp(images + delta, min=0, max=1).detach()
        
        ######################################################################
        ##  tf code #################
        ############################################################################
        images_torch = images.clone().detach().to(self.device)
        labels_torch = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images_torch, labels_torch)    
            target_labels = tf.convert_to_tensor(target_labels.numpy()).to(self.device)

        images = tf.convert_to_tensor(images.numpy())
        labels = tf.convert_to_tensor(labels.numpy())
        images = tf.cast(images, dtype=tf.float32)
        
        momentum = tf.zeros_like(images)
        v = tf.zeros_like(images)
        loss = tf.keras.losses.CategoricalCrossentropy()
 
        adv_images = tf.identity(images)

        for _ in range(self.steps):
            with tf.device('/gpu:0'):   
                with tf.GradientTape() as tape:
                    tape.watch(adv_images)
                    nes_images = adv_images + self.decay * self.alpha * momentum
                    #tape.watch(nes_images)
                    outputs = self.get_logits(nes_images)

                    if self.targeted:
                        cost = -loss(target_labels, outputs)
                    else:
                        cost = loss(labels, outputs)
            adv_grad = tape.gradient(cost, adv_images)
        
            grad = (adv_grad + v) / tf.reduce_mean(tf.abs(adv_grad + v), axis=(1, 2, 3), keepdims=True)
            grad = grad + momentum * self.decay
            momentum = grad
            
            # Calculate Gradient Variance
            GV_grad = tf.zeros_like(images)
            for _ in range(self.N):
                noise = tf.random.uniform(tf.shape(images),
                                        minval=-self.eps * self.beta,
                                        maxval=self.eps * self.beta)
                detached_adv_images = tf.identity(adv_images)
                neighbor_images = detached_adv_images + noise
                with tf.device('/gpu:0'):
                    with tf.GradientTape() as tape:
                        tape.watch(neighbor_images)
                        outputs = self.get_logits(neighbor_images)  
                        
                        if self.targeted:
                            cost = -loss(target_labels, outputs)
                        else:
                            cost = loss(labels, outputs)
                        
                GV_grad_1 = tape.gradient(cost, neighbor_images)
                GV_grad += GV_grad_1
            
            # obtaining the gradient variance
            v = GV_grad / self.N - adv_grad
            detached_adv_images = tf.identity(adv_images)
            adv_images = detached_adv_images + self.alpha* tf.sign(grad)
            delta = tf.clip_by_value(adv_images - images, clip_value_min=-self.eps, clip_value_max=self.eps)
            adv_images = tf.clip_by_value(images + delta, clip_value_min=0, clip_value_max=1)
            adv_images = tf.stop_gradient(adv_images)
        
        # Convert adv_images to a torch tensor and move it to the GPU device
        adv_images_np = adv_images.numpy()
        adv_images_torch = torch.from_numpy(adv_images_np).to(self.device)
            
            
        return adv_images_torch
