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

 Date: 31/01/2023 18:44:48
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for data_model_1
-- ----------------------------
DROP TABLE IF EXISTS `data_model_1`;
CREATE TABLE `data_model_1`  (
  `methodBody` mediumblob,
  `logList` varchar(500) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci DEFAULT NULL,
  `logNum` int(0) DEFAULT NULL,
  `methodVector` mediumtext CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci,
  `methodLocation` varchar(500) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci DEFAULT NULL,
  `seq` int(0) NOT NULL AUTO_INCREMENT,
  `methodBodyChanged` mediumblob,
  `methodLineNumber` int(0) DEFAULT NULL,
  `methodSeq` int(0) DEFAULT NULL,
  `context` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci DEFAULT NULL,
  `parentId` int(0) DEFAULT NULL,
  `logType` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci DEFAULT NULL,
  `vector` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci DEFAULT NULL,
  `fromLine` int(0) DEFAULT NULL,
  `toLine` int(0) DEFAULT NULL,
  `className` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci DEFAULT NULL,
  `methodName` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci DEFAULT NULL,
  `projectName` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci DEFAULT NULL,
  `methodParameter` mediumtext CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci,
  PRIMARY KEY (`seq`) USING BTREE,
  INDEX `data_model_1_index_1`(`logNum`) USING BTREE,
  INDEX `data_model_1_index_2`(`projectName`, `className`, `methodName`) USING BTREE,
  INDEX `data_model_1_index_3`(`methodLocation`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 581504 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

SET FOREIGN_KEY_CHECKS = 1;
