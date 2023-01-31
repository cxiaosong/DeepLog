/*
 Navicat Premium Data Transfer

 Source Server         : 本地链接
 Source Server Type    : MySQL
 Source Server Version : 80026
 Source Host           : localhost:3306
 Source Schema         : predict_log

 Target Server Type    : MySQL
 Target Server Version : 80026
 File Encoding         : 65001

 Date: 31/01/2023 18:45:23
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for project_metrix_dao
-- ----------------------------
DROP TABLE IF EXISTS `project_metrix_dao`;
CREATE TABLE `project_metrix_dao`  (
  `seq` int(0) NOT NULL AUTO_INCREMENT,
  `projectPath` varchar(255) CHARACTER SET utf8 COLLATE utf8_bin DEFAULT NULL,
  `ProjectName` varchar(255) CHARACTER SET utf8 COLLATE utf8_bin DEFAULT NULL,
  `metrix` varchar(8000) CHARACTER SET utf8 COLLATE utf8_bin DEFAULT NULL,
  `group_num` varchar(255) CHARACTER SET utf8 COLLATE utf8_bin DEFAULT NULL,
  `metrixPkg` mediumtext CHARACTER SET utf8 COLLATE utf8_bin,
  `metrixPkgVec` mediumtext CHARACTER SET utf8 COLLATE utf8_bin,
  PRIMARY KEY (`seq`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 266 CHARACTER SET = utf8 COLLATE = utf8_bin ROW_FORMAT = Dynamic;

SET FOREIGN_KEY_CHECKS = 1;
