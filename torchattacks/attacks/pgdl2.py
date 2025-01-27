import torch
import torch.nn as nn
import tensorflow as tf
from ..attack import Attack


class PGDL2(Attack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : L2

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 1.0)
        alpha (float): step size. (Default: 0.2)
        steps (int): number of steps. (Default: 10)
        random_start (bool): using random initialization of delta. (Default: True)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.PGDL2(model, eps=1.0, alpha=0.2, steps=10, random_start=True)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, device=None, eps=1.0, alpha=0.2, steps=10, random_start=True, eps_for_division=1e-10):
        super().__init__('PGDL2', model, device)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.eps_for_division = eps_for_division
        self.supported_mode = ['default', 'targeted']

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        # images = images.clone().detach().to(self.device)
        # labels = labels.clone().detach().to(self.device)

        # if self.targeted:
        #     target_labels = self.get_target_label(images, labels)

        # loss = nn.CrossEntropyLoss()

        # adv_images = images.clone().detach()
        # batch_size = len(images)

        # if self.random_start:
        #     # Starting at a uniformly random point
        #     delta = torch.empty_like(adv_images).normal_()
        #     d_flat = delta.view(adv_images.size(0), -1)
        #     n = d_flat.norm(p=2, dim=1).view(adv_images.size(0), 1, 1, 1)
        #     r = torch.zeros_like(n).uniform_(0, 1)
        #     delta *= r/n*self.eps
        #     adv_images = torch.clamp(adv_images + delta, min=0, max=1).detach()

        # for _ in range(self.steps):
        #     adv_images.requires_grad = True
        #     outputs = self.get_logits(adv_images)

        #     # Calculate loss
        #     if self.targeted:
        #         cost = -loss(outputs, target_labels)
        #     else:
        #         cost = loss(outputs, labels)

        #     # Update adversarial images
        #     grad = torch.autograd.grad(cost, adv_images,
        #                                retain_graph=False, create_graph=False)[0]
           
        #     grad_norms = torch.norm(grad.view(batch_size, -1), p=2, dim=1) + self.eps_for_division  # nopep8
        #     grad = grad / grad_norms.view(batch_size, 1, 1, 1)
        #     adv_images = adv_images.detach() + self.alpha * grad

        #     delta = adv_images - images
        #     delta_norms = torch.norm(delta.view(batch_size, -1), p=2, dim=1)
        #     factor = self.eps / delta_norms
        #     factor = torch.min(factor, torch.ones_like(delta_norms))
        #     delta = delta * factor.view(-1, 1, 1, 1)

        #     adv_images = torch.clamp(images + delta, min=0, max=1).detach()
        
        
        ###############################################
        ##  tf code #################
        ######################################
        images_torch = images.clone().detach().to(self.device)
        labels_torch = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images_torch, labels_torch)    
            target_labels = tf.convert_to_tensor(target_labels.numpy()).to(self.device)
        
        images = tf.convert_to_tensor(images.numpy())
        # images = tf.cast(images, dtype=tf.float32)
        labels = tf.convert_to_tensor(labels.numpy())
        # labels = tf.cast(labels, dtype=tf.float32)
        
            

        # Define the loss function
        loss = tf.keras.losses.CategoricalCrossentropy()

        # Create a copy of images for adversarial perturbation
        adv_images = tf.identity(images)
        # adv_images = tf.cast(adv_images, dtype=tf.float32)
        batch_size = len(images)
        
        if self.random_start:
            # Starting at a uniformly random point
            delta = tf.random.normal(shape=adv_images.shape, dtype=tf.float64)
            d_flat = tf.reshape(delta, [adv_images.shape[0], -1])
            n = tf.norm(d_flat, ord=2, axis=1)
            n = tf.reshape(n, [adv_images.shape[0], 1, 1, 1])
            r = tf.random.uniform(shape=n.shape, minval=0, maxval=1, dtype=tf.float64)
            delta *= r / n * self.eps
            adv_images = tf.clip_by_value(adv_images + delta, clip_value_min=0, clip_value_max=1)
            adv_images = tf.stop_gradient(adv_images)
      
        for _ in range(self.steps):                
            with tf.device('/gpu:0'):   
                with tf.GradientTape() as tape:
                    tape.watch(adv_images)
                    outputs = self.get_logits(adv_images)

                    # Calculate the loss
                    if self.targeted:
                        cost = -loss(outputs, target_labels)
                    else:
                        cost = loss(outputs, labels)

            # Compute gradients of the cost with respect to adv_images
            grad = tape.gradient(cost, adv_images)
            
            grad_norms = tf.norm(tf.reshape(grad, [batch_size, -1]), ord=2, axis=1) + self.eps_for_division
            grad = grad / tf.reshape(grad_norms, [batch_size, 1, 1, 1])
            adv_images = adv_images + self.alpha * grad
            
            delta = adv_images - images
            delta_norms = tf.norm(tf.reshape(delta, [batch_size, -1]), ord=2, axis=1)
            
            factor = self.eps / delta_norms
            factor = tf.minimum(factor, tf.ones_like(delta_norms))
            delta = delta * tf.reshape(factor, [-1, 1, 1, 1])
            
            adv_images = tf.clip_by_value(images + delta, clip_value_min=0, clip_value_max=1)   
            adv_images = tf.stop_gradient(adv_images)      
        
        # Convert adv_images to a torch tensor and move it to the GPU device
        adv_images_np = adv_images.numpy()
        adv_images_torch = torch.from_numpy(adv_images_np).to(self.device)
        
        return adv_images_torch
