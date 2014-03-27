
win_num = [ 1 2 4 5 6 7 ];

win_pos = zeros(4, length(win_num));

for w_idx = 1:length(win_num)
    win_pos(:,w_idx) = get(win_num(w_idx), 'Position');
end

save win.mat win_pos win_num

