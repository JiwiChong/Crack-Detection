import torch

def decorrelation_ensemble_error(true_y, model_outputs, num_experts, num_classes, optimizer, lambda_, alternate):
    decorrelation_errors_ = []
    sse_errors = []

    one_hot_target = true_y.to('cuda')

    for j in range(1, num_experts + 1):
        decorrelation_errors = []
        sse_error = [torch.pow((one_hot_target - model_outputs[j - 1]), 2)]

        for i in range(1, j + 1):
            if alternate == False:
                d = 1 if (i != j) else 0
            else:
                d = 1 if (i == j - 1) and i % 2 == 0 else 0

            P = (one_hot_target - model_outputs[i - 1].detach()) * (one_hot_target - model_outputs[j - 1])

            if d == 1:
                decorr_error_part = lambda_ * d * P
                decorrelation_errors.append(decorr_error_part)

        sse_errors.append(sse_error[0])
        decorrelation_errors_.append(decorrelation_errors)

    # If we are putting all errors into one scalar and backpropagating with it
    decorrelation_errors_ = [err[0] for err in decorrelation_errors_ if len(err) != 0]
    total_decorrelation_error = torch.sum(decorrelation_errors_[0])
    total_sse_error = torch.sum(torch.stack(sse_errors))
    total_error = total_decorrelation_error + total_sse_error

    total_error.backward()
    optimizer.step()
    optimizer.zero_grad()

    return total_error