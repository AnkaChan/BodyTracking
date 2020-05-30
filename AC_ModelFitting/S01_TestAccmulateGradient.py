import torch

if __name__ == '__main__':
    x = torch.randn(100, requires_grad=True)
    y1 = torch.randn(100, requires_grad=False)
    y2 = torch.randn(100, requires_grad=False)



    optimizer = torch.optim.Adam([x], lr=0.05)
    for i in range(1):
        optimizer.zero_grad()

        loss1 = torch.mean((y1-x)**2)/2
        loss2 = torch.mean((y2-x)**2)/2

        loss1.backward()
        loss2.backward()

        xgrad1 = x.grad.data.clone().detach()
        print(x.grad.data)

        optimizer.zero_grad()
        loss = (torch.mean((y1-x)**2) + torch.mean((y2-x)**2))/2
        loss.backward()

        xgrad2 = x.grad.data.clone().detach()

        print(x.grad.data)

        print('GradDiff:', xgrad1-xgrad2)


        optimizer.step()

        # if not i % 500:
        #     print("Loss:", loss.item())
        #     print('xy:', xy.data)
        #     print('xy**2:', (xy**2).data)
        #
        #     print('x:', x.data)
        #     print('y:', y.data)
