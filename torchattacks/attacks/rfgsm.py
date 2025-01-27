import torch
import torch.nn as nn
import tensorflow as tf
from ..attack import Attack


class RFGSM(Attack):
    r"""
    R+FGSM in the paper 'Ensemble Adversarial Training : Attacks and Defences'
    [https://arxiv.org/abs/1705.07204]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): strength of the attack or maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 10)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.RFGSM(model, eps=8/255, alpha=2/255, steps=10)
        >>> adv_images = attack(images, labels)
    """

    def __init__(self, model, device=None, eps=8/255, alpha=2/255, steps=10):
        super().__init__('RFGSM', model, device)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
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

        # adv_images = images + self.alpha*torch.randn_like(images).sign()
        # adv_images = torch.clamp(adv_images, min=0, max=1).detach()

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
        loss = tf.keras.losses.CategoricalCrossentropy()

        # Assuming 'adv_images' and 'images' are TensorFlow tensors, and 'self.alpha' is a scalar value
        adv_images = images + self.alpha * tf.sign(tf.random.normal(tf.shape(images), dtype=images.dtype))
        adv_images = tf.clip_by_value(adv_images, clip_value_min=0, clip_value_max=1)
        adv_images = tf.stop_gradient(adv_images)  # Equivalent to torch.detach()
        
        adv_images = tf.cast(adv_images, dtype=tf.float32)
        adv_images = tf.Variable(adv_images, trainable=True)
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
            
            grad = tape.gradient(cost, adv_images)
            adv_images = tf.stop_gradient(adv_images) + self.alpha * tf.sign(grad)
            images = tf.cast(images, dtype=tf.float32)
            delta = tf.clip_by_value(adv_images - images, clip_value_min=-self.eps, clip_value_max=self.eps)
            adv_images = tf.clip_by_value(images + delta, clip_value_min=0, clip_value_max=1)
            adv_images = tf.stop_gradient(adv_images)
                   
        # Convert adv_images to a torch tensor and move it to the GPU device
        adv_images_np = adv_images.numpy()
        adv_images_torch = torch.from_numpy(adv_images_np).to(self.device) 
        
        return adv_images_torch
