import torch
import torch.nn as nn
import tensorflow as tf

from ..attack import Attack


class SINIFGSM(Attack):
    r"""
    SI-NI-FGSM in the paper 'NESTEROV ACCELERATED GRADIENT AND SCALEINVARIANCE FOR ADVERSARIAL ATTACKS'
    [https://arxiv.org/abs/1908.06281], Published as a conference paper at ICLR 2020
    Modified from "https://githuba.com/JHL-HUST/SI-NI-FGSM"

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of iterations. (Default: 10)
        decay (float): momentum factor. (Default: 1.0)
        m (int): number of scale copies. (Default: 5)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.SINIFGSM(model, eps=8/255, alpha=2/255, steps=10, decay=1.0, m=5)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, device=None, eps=8/255, alpha=2/255, steps=10, decay=1.0, m=5):
        super().__init__('SINIFGSM', model, device)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.m = m
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

        # loss = nn.CrossEntropyLoss()

        # adv_images = images.clone().detach()

        # for _ in range(self.steps):
        #     adv_images.requires_grad = True
        #     nes_image = adv_images + self.decay*self.alpha*momentum
        #     # Calculate sum the gradients over the scale copies of the input image
        #     adv_grad = torch.zeros_like(images).detach().to(self.device)
        #     for i in torch.arange(self.m):
        #         nes_images = nes_image / torch.pow(2, i)
        #         outputs = self.get_logits(nes_images)
        #         # Calculate loss
        #         if self.targeted:
        #             cost = -loss(outputs, target_labels)
        #         else:
        #             cost = loss(outputs, labels)
        #         adv_grad += torch.autograd.grad(cost, adv_images,
        #                                         retain_graph=False, create_graph=False)[0]
        #     adv_grad = adv_grad / self.m

        #     # Update adversarial images
        #     grad = self.decay*momentum + adv_grad / \
        #         torch.mean(torch.abs(adv_grad), dim=(1, 2, 3), keepdim=True)
        #     momentum = grad
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
        loss = tf.keras.losses.CategoricalCrossentropy()

        adv_images = tf.identity(images)
        
        for _ in range(self.steps): 
            adv_grad = tf.zeros_like(images)
            with tf.device('/gpu:0'):         
                for i in tf.range(self.m):
                    with tf.GradientTape() as tape:
                        tape.watch(adv_images)
                        nes_image = adv_images + self.decay * self.alpha * momentum 
                        nes_images = nes_image / tf.pow(2.0, tf.cast(i, tf.float32))
                        outputs = self.get_logits(nes_images)
                        if self.targeted:
                            cost = -loss(target_labels, outputs)
                        else:
                            cost = loss(labels, outputs)
                
                    # Compute gradients of the cost with respect to adv_images
                    grad = tape.gradient(cost, adv_images)
                    adv_grad += grad
                        
                
            adv_grad = adv_grad / self.m

            grad = self.decay * momentum + adv_grad / \
                tf.reduce_mean(tf.abs(adv_grad), axis=(1, 2, 3), keepdims=True)
            momentum = grad
            adv_images = tf.stop_gradient(adv_images) + self.alpha * tf.sign(grad)
            delta = tf.clip_by_value(adv_images - images, clip_value_min=-self.eps, clip_value_max=self.eps)
            adv_images = tf.clip_by_value(images + delta, clip_value_min=0.0, clip_value_max=1.0)
        
        # Convert adv_images to a torch tensor and move it to the GPU device
        adv_images_np = adv_images.numpy()
        adv_images_torch = torch.from_numpy(adv_images_np).to(self.device)        
        return adv_images_torch
