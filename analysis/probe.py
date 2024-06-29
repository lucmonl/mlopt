from sklearn.linear_model import LogisticRegression

def create_keys(graphs, zero_out_top, zero_out_selfattn, zero_out_attn):
    if zero_out_attn == -1 and zero_out_attn not in graphs:
        graphs[zero_out_attn] = []
        return

    if zero_out_top not in graphs:
        graphs[zero_out_top] = {}

    if zero_out_selfattn not in graphs[zero_out_top]:
        graphs[zero_out_top][zero_out_selfattn] = {}
    graphs[zero_out_top][zero_out_selfattn][zero_out_attn] = []

def transformer_probe(graphs, model, train_loader, test_loader, device, **kwargs):
    zero_out_attn, zero_out_top, zero_out_selfattn = kwargs["zero_out_attn"], kwargs["zero_out_top"], kwargs["zero_out_selfattn"]
    create_keys(graphs.layer_cls_train_score, zero_out_top, zero_out_selfattn, zero_out_attn)
    create_keys(graphs.layer_cls_test_score, zero_out_top, zero_out_selfattn, zero_out_attn)

    logisticRegr = LogisticRegression(max_iter=100)
    
    cls_tokens, targets = model.get_cls_tokens(train_loader, device, zero_out_attn, zero_out_top, zero_out_selfattn)
    cls_tokens_test, targets_test = model.get_cls_tokens(test_loader, device, zero_out_attn, zero_out_top, zero_out_selfattn)
    #graphs.layer_cls_train_score[zero_out_top][zero_out_selfattn][zero_out_attn] = []
    #graphs.layer_cls_test_score[zero_out_top][zero_out_selfattn][zero_out_attn] = []
    for layer in range(len(cls_tokens)):
        
        x_train, y_train =cls_tokens[layer], targets
        logisticRegr.fit(x_train, y_train)
        train_score = logisticRegr.score(x_train, y_train)

        x_test, y_test =cls_tokens_test[layer], targets_test
        test_score = logisticRegr.score(x_test, y_test)
        print(layer, train_score, test_score)
        if zero_out_attn != -1:
            graphs.layer_cls_train_score[zero_out_top][zero_out_selfattn][zero_out_attn].append(train_score)
            graphs.layer_cls_test_score[zero_out_top][zero_out_selfattn][zero_out_attn].append(test_score)
        else:
            graphs.layer_cls_train_score[zero_out_attn].append(train_score)
            graphs.layer_cls_test_score[zero_out_attn].append(test_score)